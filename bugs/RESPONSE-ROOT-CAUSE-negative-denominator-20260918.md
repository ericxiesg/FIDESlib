# 根因找到了：自举把 padding 槽的 0 变成 ±0.017，**负的那一半**让 Goldschmidt 发散

日期：2026-09-18。承接 `6885c31`。**已修，本机复现 + 验证。**

---

## 0. 结论

你的 `noise_level` 全=1、断言不触发、组合测试 49 passed——**这些都对，因为问题不在算子**。
问题在**输入的符号**。

线索其实一直在仓库里：`test_bootstrap_noise_level.py` 的 xfail 写着

> bootstrap precision is only ~10.9 bits (**max abs error 0.017**) at dnum=4 …
> **This low precision overwhelms he_inv's 2e-4 denominator.**

而 `he_inv` 第一件事就是 `bootstrap(denominator)`。

padding 行的分母**精确等于 0**（`_carrying` 的 docstring 早就写了这点）。
自举之后，这些槽变成 **±0.017**——**一半是负数**。
Goldschmidt 从正的 `ones` 出发，对负的 `x` 求 `1/x` **没有收敛点**，只会每轮变大。

### 本机注入 0.017 有符号误差，逐位复现你的现象

| 自举误差 | iter03_b | iter04_b | iter05_b |
|---|---|---|---|
| 0（干净） | 0.003888 | 0.003902 | 0.003903 |
| **0.017 有符号** | **0.05821** | **3.521** | **1.023e+04** |
| 0.017 **去掉符号**（取绝对值） | 0.003888 | 0.003903 | 0.003903 |
| 0.05 有符号 | 1.001 | 978.4 | 7.888e+08 |

**同样的幅度，去掉符号就完全干净。** 是符号，不是幅度。
发散从 **iter03 的 b** 开始，和你的探针（iter03_b=241.8 → iter04 2.4e7 → iter05 1.6e17）同一个形状。

### 为什么 `b` 先于 `a`

`b` 是分母那一支，直接吃到负数。`a` 从 `ones` 出发，在 padding 槽上≈0，所以它本身不动——
但 `correction = 标量 − b` 被 `b` 带坏之后，`a ← a·correction` 每轮乘一个巨大的数，
于是下一轮 `a` 也炸。这正好解释你看到的"b 先 a 后"。

---

## 1. 修法：迭代前把支撑集外的槽抬到 0.5

```python
inv_D, ... = self.he_inv(total, self._carrying(attention_mask),
                         epsilon=..., alpha=..., support=self._support(attention_mask))
```

`support` 是**明文**指示器，`he_inv` 在**自举之后**做一次明文加法：

```python
refreshed = self.add(refreshed, PADDING_FLOOR * (1.0 - support))
```

明文加法**不花 level**。`he_invsqrt` 同样处理（它本来就收 `mask` 这个明文指示器）。

### 为什么是 0.5 而不是 1.0

我第一版写的 1.0，**不管用**——因为同样的误差也落在填充值上，1.0 变成 1.017，
**跑到 `[epsilon, 1]` 上界外面**，于是从上面发散。实测：

| 填充值 | iter05_b |
|---|---|
| 不填 | 1.023e+04 |
| **1.0** | **2.02e+04**（还是炸） |
| 0.25 / 0.5 / 0.75 | **0.003903**（和干净时逐位相同） |

取中点，离两端都最远。这条写进了 `PADDING_FLOOR` 的注释和一个专门的测试，
免得一年后有人把它"优化"回 1.0。

---

## 2. 这只治发散，不治精度——请分开看

修好之后 `b` 不再爆炸，但**活跃槽上的误差仍然在**：

| 自举误差 | 活跃槽最差相对误差（已带 floor） |
|---|---|
| 0.017 | **0.58** |
| 0.05 | 2.6e14 |

0.017 的绝对误差打在 0.05 量级的分母上就是 34% 的相对误差，**没有任何 filler 能救**。
这一半是**自举精度**问题，就是 `test_bootstrap_noise_level.py` 里记着的 10.9 bit vs 应有的 20–25 bit。

所以预期：**这次修完，`07d` / `11b` 不会再出 1e16 / 1e124，但 `hidden` 未必立刻就准**。
如果修完 MAE 从 2.4e19 降到 O(1) 但还不到 1e-3，那就是自举精度，下一步是 Meta-BTS
（`bootstrap_twice` 已经在 `stages.py`，而且按 `--time-ops` 只要 +0.76 s/层，代价可以忽略）。

---

## 3. 验证（本机，无 CUDA）

- `tests/test_division_padding_floor.py` 3 个用例：不填会发散（>1e3）、填了不发散（<1e2）、
  **填 1.0 又会发散**（守住那个反直觉的选择）。
- 注意测试读的是**逐轮探针**，不是返回值：发散在 `b` 里，而返回值被 `ones` 关住，
  **从外面看是正常的**——这也是这个 bug 活这么久的原因。
- `bench fhe --engine clear`：`relRMSE 2.776e-03`，**和修之前逐位相同**（clear 上 padding 本来就是精确 0，
  填充只是把 0 换成 0.5，而 `ones` 在那里是 0，结果不变）。

---

## 4. 请在设备上

同一条命令重跑，`--per-stage --time-ops`：

1. `07d.inverse_denominator` 的 **max** 应该从 5.3e16 掉到 O(1)。
2. `11b.inverse_sqrt` 的 max 应该从 1.87e124 掉到 O(1)。
3. `11a.variance` 应该回到 0.2 量级（它之前是被 `10.attention_dense` 的 1.3e19 带坏的）。
4. `hidden` MAE——**报回来就行，不一定马上好**，见 §2。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
