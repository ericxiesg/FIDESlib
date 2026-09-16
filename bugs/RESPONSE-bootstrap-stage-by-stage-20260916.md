# 分阶段定位：Stage 4 (SlotsToCoeffs) 把 9.5e-5 放大到 0.016

日期：2026-09-16。针对 `fb5d499`。

---

## 实验

用 `bootstrap_stage(ct, stage)` 对零密文分阶段 bootstrap 并解密。
每个阶段的输出应该接近零；第一个不接近零的就是误差来源。

## 零密文结果

```
input                          max 7.88e-10
after stage 1 (ModRaise+scale)  max 8.23e-02   p50 5.45e-03   p99 3.53e-02
after stage 2 (CoeffsToSlots)   max 3.21e-01   p50 5.36e-02   p99 1.79e-01
after stage 3 (mod reduction)   max 9.47e-05   p50 8.40e-06   p99 9.47e-05
after stage 4 (SlotsToCoeffs)   max 1.58e-02   p50 2.61e-03   p99 1.00e-02
whole bootstrap                 max 6.31e-02   p50 1.04e-02   p99 4.00e-02
```

## 非零密文 [2, -3, 0.5, 0, ...]

```
input                          max 3.00e+00
after stage 1 (ModRaise+scale)  max 9.37e-02   p50 5.50e-03
after stage 2 (CoeffsToSlots)   max 2.86e-01   p50 5.36e-02
after stage 3 (mod reduction)   max 1.31e-04   p50 1.90e-05
after stage 4 (SlotsToCoeffs)   max 7.51e-01   p50 2.63e-03
```

---

## 分析

### 关键转折在 Stage 3 → Stage 4

```
Stage 3 (mod reduction):  max 9.47e-05    ← 非常干净
Stage 4 (SlotsToCoeffs):  max 1.58e-02    ← 放大了 167x
```

**Stage 3 输出几乎为零（9.5e-5），Stage 4 把它放大到 0.016。**
SlotsToCoeffs（StC）是最后一个线性变换，负责把系数域转回 slot 域。

### Stage 1-2 的噪声是中间格式

Stage 1 (ModRaise + scale) 的 0.08 和 Stage 2 (CtS) 的 0.32 看起来大，
但 Stage 2 的输出是**系数域**（不是 slot 域），值域和单位不同。
而且 Stage 3 把它们压到了 9.5e-5 — 说明 Stage 1-2 的"噪声"在模约简之后被消除了，
不是真正的误差。

### 为什么 Stage 4 放大了 167x

StC 和 CtS 是同一类线性变换（DFT 类），但方向相反。如果 CtS 在 Stage 2 产生 0.32
的系数域噪声，经过 Stage 3 的模约简后只剩 9.5e-5，那 Stage 4 的 StC **重新引入**
了类似量级的噪声——而不是放大 Stage 3 的输出。

**StC 自己就在注入噪声。** 和 CtS 一样，它是明文乘 + 旋转的组合。
但单次旋转只有 4e-10（实验 2 测过），4000 次累积也才 2e-6。
StC 的 0.016 远超旋转累积能解释的范围。

### 候选根因

1. **StC 对角线明文编码精度**：对角线在某个 level 编码，编码时的舍入误差被乘到密文上。
   如果对角线编码精度不够（比如在太低 level 编码），每次明文乘都带入误差。
2. **StC 的 level 对齐**：虽然 CtS 的 `alignToDiagonals` 诊断没触发，
   但 StC 可能有自己的对齐逻辑（`Bootstrap.cu:313-339` 的 `EvalSlotsToCoeffs`/
   `EvalLinearTransform`），需要单独检查。
3. **`constantEvalMult` 缩放**：协作者 §4 提到的 `Bootstrap.cu:250` 的
   `constantEvalMult = pre/(k*N)` 在 Stage 1 执行。Stage 3 模约简后值很小，
   Stage 4 的 StC 可能在错误的 scale 上操作。

---

## 下一步

1. **在 StC 内部加 `stopAfterStage` 式的探针**：StC 本身是多步线性变换
  （level budget 3 意味着 3 步），在每步后解密看噪声在哪一步引入。
2. **打印 `constantEvalMult` 和 `corFactor`**：确认缩放/放大倍数。
3. **检查 StC 的对角线编码 level**：和 CtS 一样，确认每步对角线 level 一致。

---

## 测试

210 passed, 13 skipped（含 `test_which_bootstrap_stage_introduces_the_error`，
CPU 下 skip bench params 的）。
