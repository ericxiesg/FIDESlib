# `he_invsqrt` / `refresh` 的 GPU 实现核查：一个只有 LayerNorm 走、而且没有任何测试的分支

日期：2026-09-18。承接 `9ed3b91`（发散点 = norm_1）。

---

## 0. 结论

`he_invsqrt` 和 `refresh` 都是 Python 组合，没有独立的 CUDA 实现。逐个原语核查后，
**只有一条分支是它们独有且未被任何测试覆盖的**：

```python
# pyfideslib/__init__.py:147-152
def subtract(self, x, y):
    if isinstance(x, (int, float)):
        return self.cc.EvalScalarSub(float(x), y)          # <- he_inv 走这条，softmax 正常
    if isinstance(x, np.ndarray):
        pt = self.encode(x, level=self.depth - self.level(y))
        return self.cc.EvalNegate(self.cc.EvalSubPt(y, pt))  # <- he_invsqrt 走这条
```

`numeric.py:352` 的 `self.subtract((3 / k) * mask, a)` 是**全仓库唯一**一处把
numpy 数组作为 `subtract` 左操作数的调用。`he_invsqrt` 只被 LayerNorm 调用，
而 LayerNorm 正是 `--per-stage` 定位到的发散起点。

---

## 1. 两个函数用到的原语

### `he_invsqrt`（`numeric.py:333`）

| 原语 | 上游是否也用 |
|---|---|
| `subtract(ndarray, ct)` | **否，全仓库仅此一处** |
| `multiply(ct, ndarray)` + rescale | 是 |
| `align` / `relinearize` / `square` / ct×ct | 是 |

### `refresh`（`layernorm.py:186`）

| 原语 | 上游是否也用 |
|---|---|
| `multiply_1j` | 是（stage 07） |
| `conjugate` | 是（stage 07） |
| `multiply(ct, float)` + rescale | 是 |
| `multiply(ct, int)`（level-free） | 是（stage 07 的 `score_refresh_scale`） |
| `bootstrap` | 是 |
| `add` / `subtract(ct, ct)` | 是 |

**`refresh` 没有独有原语**，它用的每一样 stage 07 都用过，而 stage 07 在设备上是干净的
（`07a`–`07d` 和我这边的 ClearEngine 对到 3–4 位有效数字）。

---

## 2. 测试覆盖

`tests/test_stage2_linear.py:14-16`：

```python
engine.subtract(cx, cy)      # ct - ct        ✅
engine.subtract(2.0, cx)     # scalar - ct    ✅  <- EvalScalarSub 分支
```

**没有任何测试传 ndarray 作为左操作数。** `EvalNegate(EvalSubPt(...))` 这条路
在两个 engine 上都从没被单独验证过。

已加 `test_subtract_a_ciphertext_from_a_plaintext_vector`，用 `he_invsqrt` 实际用的形状
（稀疏 mask 作左操作数），并且单独断言**掩码之外的符号**——那里的正确结果是 `-x`，
符号错的话正是"LayerNorm 返回 -1.5e107"的样子。

本机 `pyfideslib` 没编译，测试 skip；**请在你那边跑**：

```
python -m pytest tests/test_stage2_linear.py -q
```

---

## 3. 数学上这条分支是对的，要查的是元数据

`EvalNegate(EvalSubPt(y, pt))` = `-(y - pt)` = `pt - y` ✅。ClearEngine 上实测误差 0。

所以如果它坏，坏的不是代数，而是下面之一：

1. **`EvalNegate` 是否保留 scale degree / NoiseLevel**。`he_invsqrt` 之后紧接着
   `align` + ct×ct，FIXEDMANUAL 下 degree 不对会在后面炸而不是当场炸。
2. **`encode(x, level=self.depth - self.level(y))` 的 level 约定**。同一个表达式在
   `ct - ndarray` 分支（:156）也用，但那条被大量使用且正常——所以更可能是 (1)。
3. `a` 进 `subtract` 时的 degree：`he_invsqrt` 里 `a` 来自
   `rescale(relinearize(multiply(...)))`，应该是 canonical，但值得在设备上确认。

---

## 4. 另一个已加的东西：`he_invsqrt` 现在有范围检查

`he_inv` 有 `_check_inversion_range`（这一轮它抓到三个 bug）。**`he_invsqrt` 一直没有。**
已补 `_check_invsqrt_range`：方差跑出 `[min_var/max_var, 1]` 就抛，并且报出实际区间。

这能把"窗口不对"和"算术不对"分开：
- 抛异常 → 方差离开了 `VARIANT_BOUNDS`，是标定问题；
- 不抛但设备仍炸 → 是上面第 3 节的元数据问题。

`stage_11_attention_layernorm` 用 variant 1，窗口 `[0.15/10.0, 1] = [1.5e-02, 1]`。

---

## 5. 我核查过但**排除**的

| 假设 | 结论 |
|---|---|
| `variance = n*Σx² − (Σx)²` 灾难性相消 | **排除**。真实 checkpoint 上条件数 **1.01**（mean²/var ≈ 0.005），不相消 |
| bootstrap 精度 | 你已排除（15.2 bit），我这边 noise model 也不炸 |
| softmax 链 | 排除（`07a`–`07d` 正常，且和我的 ClearEngine 对得上） |
| `residual_scale` 折叠 | 排除（整层等价性测试 1e-9 通过） |
| key-switch 噪声 | 排除（一层累积 8.2e-11 : bootstrap 4.9e-04 = 1/5941767） |

---

## 6. 建议的下一步

1. 跑 `tests/test_stage2_linear.py`——如果新测试红了，直接结案。
2. 如果绿了，在 `norm_1` 内部按你 §5 的计划加探针，但**先看 `he_invsqrt` 的范围检查抛不抛**
   （`--engine clear` 跑一次就知道，不用 GPU）。
3. 设备上确认 `EvalNegate` 之后的 `GetNoiseLevel()` / degree。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
