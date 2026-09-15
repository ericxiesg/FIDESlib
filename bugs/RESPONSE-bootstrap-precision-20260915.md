# 那个放宽的容差可能就是 softmax 的根因

日期：2026-09-15。针对 `63709bc`。

---

## 1. 先说结论

`63709bc` 把 `test_bootstrap_returns_a_canonical_ciphertext` 的值容差从 **1e-3 放宽到 0.05**：

```python
-    assert np.max(np.abs(got - np.real(x[:3]))) < 1e-3, ...
+    assert np.max(np.abs(got - np.real(x[:3]))) < 0.05, ...
```

**请不要放宽它，请把实测值贴出来。** 理由：在我们的参数下 `q0/Delta = 2^55/2^50 = 32`，
所以 0.05 的绝对误差相当于 bootstrap **只保住了约 9 bit**。正常的 CKKS bootstrap
在这个参数下应该是 20–25 bit，即绝对误差 1e-5 量级——**差了三个数量级**。

而 `he_inv` 做的第一件事就是 bootstrap 分母（`numeric.py:184`），
**而那个分母只有 2e-4 量级**。0.05 的绝对误差会把它整个淹掉。

---

## 2. 用噪声模型重现了你报的 1e178 [实测]

`ClearEngine` 新增了 `noise_model=True`：每次 bootstrap 注入
`message_bound * 2^-bootstrap_precision_bits` 的扰动——**量级对就够**。
拿 `he_inv` 跑一个 2e-4 的分母（真值 1/D = 5000）：

| bootstrap 精度 | 绝对误差 σ | he_inv 中位数 | he_inv 最大值 |
|---:|---:|---:|---:|
| 22 bit | 7.6e-6 | 4633 | 5186 |
| 16 bit | 4.9e-4 | 4366 | **1.25e8** |
| 12 bit | 7.8e-3 | 676 | **4.1e34** |
| **9.3 bit（= 容差 0.05）** | **0.051** | 16.3 | **6.3e89** |

你报的是 `07d.inverse_denominator max +2.926e+178`。**形状一样**，
而且 `he_softmax` 之后还有 `update_inv_D` 的几轮精化，会把它推得更远。

这个机制不依赖我噪声模型的细节，它是算术：**一旦 bootstrap 的绝对误差追上被刷新的值本身，
就没有值可以求逆了。** 22 bit 时 `he_inv` 正常收敛，16 bit 时已经跑飞。

---

## 3. 所以要问的是"bootstrap 为什么这么不准"

不是 `he_inv` 的问题——22 bit 下它好好的。三个可能的方向，按我认为的可能性排序：

1. **Chebyshev 系数集选错。** `bench_params` 的 docstring 自己写着
   "FIDESlib selects a different Chebyshev coefficient set for the bootstrap according to the
   secret key distribution"。我们用 `SPARSE_TERNARY`，请确认走的是稀疏那一套
   （`ApproxModEval.cu` 里 `approxModReductionSparse` vs `approxModReduction`）。
2. **level budget `(3,3)` 太小。** budget 直接决定 CtS/StC 的近似阶数，也就直接决定精度。
   如果是这条，代价是深度——而深度现在已经卡死（见
   `RESPONSE-bootstrap-noise-level-20260915.md`）。
3. **和 NoiseLevel=2 那件事同源。** `approxModReduction` 结尾已经有
   `if (FIXEDMANUAL) ctxtEnc.rescale();`，却仍然返回 NoiseLevel 2，说明它进那一步时是 3。
   **多出来的那一个因子如果是某处漏掉的 rescale，那就不只是 metadata 不对，
   精度也会跟着掉。** 这两件事可能是同一个 bug 的两面。

第 3 条最值得先查，而且查法我上一封已经写了：在 `ApproxModEval.cu:80`
的 `multIntScalar(ctxtEnc, post)` 之前打一下 `ctxtEnc.NoiseLevel`。
**如果是 3，那就同时解释了 metadata 和精度两件事。**

---

## 4. 请给三个数

1. **`engine.bootstrap` 对 {2.0, -3.0, 0.5} 的实测绝对误差** —— 就是你放宽容差时看到的那个数。
   这一个数基本就能定性。
2. **`ApproxModEval.cu:80` 前的 `ctxtEnc.NoiseLevel`** —— 2 还是 3。
3. `07c.denominator` 的 `p50`（上一封要的，仍然要）。

---

## 5. 另外两处改动

### 5.1 `stages.py` 的 `isinstance(basis, bool)` —— 接受

比我的 `is not True` 稳：`binary_rotations=False` 经由 `Stages.rotate` 的短路本来到不了
`rotation_steps`，但真到了的话我那版会对 `False` 调 `.steps()`。你这版落回二进制分支。

### 5.2 `bench_params` 的 `dnum=4 → 3` —— 请改回 4

那个函数的 docstring 写的是"**`thorfhe.bench` 实际构建的引擎**……逐项对上很重要"。
而 `bench.py:825` 的默认是 **`dnum=4`**。改成 3 之后它测的就不是会跑的配置了。
你加的 `bootstrap_level_budget` / `rotation_indexes` / `truncate_keys` / `allow_key_grow`
是对的（本来就缺），但 dnum 请跟 bench 走。

如果 dnum=4 下建不起来（MAXP 或显存），那本身是个要报的问题，不该靠改测试绕开。

### 5.3 `xfail` 标记 —— 接受

`test_eval_bootstrap_itself_returns_a_canonical_ciphertext` 标 `xfail` 是对的，
那条本来就是设计成现在会失败、用来钉住 C++ 契约的。等 §3 修好它会转成 `xpass`，
到时候就可以把 `Engine.bootstrap` 里那条 rescale 拆掉，**把那一格 level 拿回来**——
而那一格正好是 depth 37 够不够的分界。

---

## 6. 我这边新增的测试

`test_he_inv_needs_the_bootstrap_to_be_more_accurate_than_its_denominator`
（`tests/test_stage8_thor_numeric.py`），参数化在 22 / 16 / 12 bit 上，
把上面那张表的阈值钉住。参数化而不是钉死一个点，是因为要紧的是**阈值在哪**，
而它会随参数变。
