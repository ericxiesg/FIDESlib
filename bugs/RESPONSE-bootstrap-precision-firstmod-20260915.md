# first_mod_bits 实验：绝对误差恒定，根因在线性变换 (CtS/StC)

日期：2026-09-15。针对 `78729d3` 中 §5 建议的实验。

---

## 实验

按协作者建议，只改 `first_mod_bits`（55→58），其余不动，测 bootstrap 对
{2.0, -3.0, 0.5} 的绝对误差。判据是二选一：

- 误差随 `q0/Delta` 缩放 → 模约简部分（Chebyshev / double-angle / correctionFactor）
- 误差基本不变 → 线性变换部分（CtS / StC）

## 结果

```
fmb=55: q0/Delta=  32, max_error=2.27e-02, 10.5 bits
fmb=56: q0/Delta=  64, max_error=1.68e-02, 11.9 bits
fmb=57: q0/Delta= 128, max_error=2.22e-02, 12.5 bits
fmb=58: q0/Delta= 256, NaN (bootstrap crashes)
```

**绝对误差在 0.017–0.023 范围内基本恒定**，`q0/Delta` 从 32 翻到 128（4x）
误差没有随之放大。精度位数微涨（10.5→12.5）只是因为分母 `q0/Delta` 变大了，
绝对误差本身没有跟。

## 结论

**根因在线性变换部分（CtS / StC），不在模约简部分。**

如果误差来自 Chebyshev/double-angle，`q0/Delta` 翻 4 倍误差应该翻到 ~0.09。
实际误差反而略降（0.023→0.022），与"随 `q0/Delta` 缩放"的预测完全不符。

## 两个附带观察

1. **fmb=58 (q0/Delta=256) 产生 NaN**：ModRaise 留下 `m + q0*I`，正弦近似只在
   零附近有效。q0/Delta=256 时消息值 2.0, 3.0 可能已经超出近似范围，
   或者 depth=37 + fmb=58 的参数组合让某个 level 算术溢出。

2. **精度从 10.5 涨到 12.5 不是真改善**：多出来的 2 bit 来自精度公式
   `-log2(error / q0_Delta)` 的分母变大，不是绝对误差变小。对 `he_inv`
   的 2e-4 分母来说，0.02 的绝对误差仍然是 100 倍，仍然完全不可用。

## 下一步

往 CtS / StC 查。具体方向：

1. **`EvalCoeffsToSlots` 的对角线对齐**：协作者之前提到它"对齐对角线多花 1 level"，
   如果对齐出错（比如对角线在错误的 level 上编码），StC 的明文乘会引入额外噪声。
2. **`EvalLinearTransform` 的预计算对角线**：`Bootstrap.cu:213` 的
   `sparse_encaps` 控制走 `EvalLinearTransform` 还是 `EvalCoeffsToSlots`，
   我们的配置 `N/2 == slots` 走的是非 sparse 路径。
3. **correctionFactor = 0**：我们传给 `EvalBootstrapSetup` 的是 `[0, 0]`，
   `Bootstrap.cu:204` 的 `correction = correctionFactor - deg` 在 0 时是负数，
   `corFactor = 1 << llround(correction)` 会产生奇怪值。值得打印确认。

## 测试修复

`test_polynomial_reproducer.py::_float_scalar_on_degree_two`：协作者正确指出
`Ciphertext::multScalar` 在 `Ciphertext.cpp:913` 有 `assert(NoiseLevel == 1)`，
所以 float scalar 在 degree-2 上是非法的（ClearEngine 的 guard 是对的）。
修复：在 multiply 之前先 rescale（degree-2 → scale Δ^1），测试通过。

208 passed, 7 skipped。
