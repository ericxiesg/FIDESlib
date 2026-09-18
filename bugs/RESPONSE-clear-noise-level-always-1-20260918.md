# ClearEngine 确认：he_invsqrt 的 noise_level 始终为 1

日期：2026-09-18。针对 `1d70d95`。

---

## 1. 你说得对：我的合成测试不是流水线路径

我 §2 的复现是 encrypt 后**连续 rescale 没有乘法**，流水线不这么做。
每次 rescale 前都有乘法（1→2→1），noise_level 始终为 1。

---

## 2. ClearEngine 实测：he_invsqrt 全程 noise_level = 1

在 ClearEngine 上 patch `he_invsqrt`，记录每次 `subtract(ndarray, ct)` 时
密文的 level 和 noise_level：

```
he_invsqrt #1 (attention layernorm):
  entry    level=17  noise_level=1
  subtract  level=17  noise_level=1
  subtract  level=15  noise_level=1
  subtract  level=13  noise_level=1
  subtract  level=11  noise_level=1
  subtract  level= 9  noise_level=1
  exit     level= 8  noise_level=1

he_invsqrt #2 (output layernorm):
  entry    level=14  noise_level=1
  subtract  level=14  noise_level=1
  subtract  level=12  noise_level=1
  subtract  level=10  noise_level=1
  subtract  level= 8  noise_level=1
  subtract  level= 6  noise_level=1
  subtract  level= 4  noise_level=1
  exit     level= 3  noise_level=1
```

**全部 noise_level = 1。** 没有负数，没有下溢。

---

## 3. 回答你的问题

> §1 的错误信息是真实那次 layer 0 跑出来的，还是 §2 合成测试跑出来的？

**是合成测试。** 我手动 encrypt 后连续 rescale 到低 level，这不是流水线的真实路径。
真实流水线里 noise_level 始终为 1（ClearEngine 证明了）。

---

## 4. 那么 norm_1 的发散原因还没找到

排除项：
- ✅ noise_level 变负（合成测试假象，真实跑里始终为 1）
- ✅ bootstrap 精度（15.2 bit，noise model 不炸）
- ✅ softmax 链（07a-07d 正常）
- ✅ subtract(ndarray, ct) 在高 level 正常（GPU 实测 1e-13）
- ✅ key-switch 噪声（8.2e-11 vs bootstrap 4.9e-4）

**还没排除的**：
- subtract(ndarray, ct) 在**低 level**（level 9-17）的 GPU 行为——
  我在 level 37 测了正常，但 he_invsqrt 在 level 9-17 调用它。
  需要在 GPU 真实跑里确认。
- `he_invsqrt` 的 Goldschmidt 迭代在 GPU 上的累积舍入
- `refresh` 的 bootstrap 在链底（低 level）时精度下降

---

## 5. 正在跑

GPU 端到端带 `--per-stage` + subtract 错误 catch，正在跑。
由于 sb=59 性能不稳定（有时 19min 有时 90min+），结果可能需要等。

如果 subtract(ndarray, ct) 在 GPU 真实跑里**不报错**，
那发散原因在别处——可能是 Goldschmidt 迭代的 GPU 舍入累积。
