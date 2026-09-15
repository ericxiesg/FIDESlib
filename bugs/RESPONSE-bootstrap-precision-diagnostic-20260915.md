# 诊断结果：对角线 level 一致，底噪 0.01-0.06，周期 2048 结构

日期：2026-09-15。针对 `17f3065`。

---

## 1. 对角线 level 诊断 — 清白

`CoeffsToSlots.cu` 的 `alignToDiagonals` 诊断打印**没有触发**。三次 bootstrap
（3 values、零密文、全 1.0）均无 `FIDESlib: the diagonals ... not all at one level`
warning。

**同一步内所有对角线 level 一致**，`alignToDiagonals` 取 min 的假设成立。
这处代码清白，不是精度问题的根因。

---

## 2. 零密文实验 — 底噪 0.01-0.06

```
=== Test 2: zero ciphertext ===
  max=6.186e-02  p50=1.044e-02  p99=3.959e-02
```

**零消息的 bootstrap 底噪就有 0.01-0.06。** 这不是消息驱动的误差——
没有值可以让误差"成比例"。是某处在**加**东西。

按协作者 §6.1 的判据：**底噪 ~0.02 → 误差与消息完全无关，是某处在加东西。**

---

## 3. 全 1.0 实验 — 误差和零密文几乎一样

```
=== Test 3: all 1.0 ===
  max=6.745e-02  p50=1.046e-02  p99=4.026e-02
```

对比零密文：max 0.062→0.067，p50 0.010→0.010，p99 0.040→0.040。
**填入 1.0 的消息几乎不改变误差分布。** 误差不是消息的函数。

这和 §2 一致：误差是加性的，不是乘性的。

---

## 4. 误差结构 — 周期 2048

```
  slot%16:   mean range [1.21e-02, 1.27e-02], spread=6.55e-04
  slot%128:  mean range [1.09e-02, 1.38e-02], spread=2.91e-03
  slot%2048: mean range [5.72e-03, 2.08e-02], spread=1.51e-02
```

按协作者 §6.2 的判据：
- **误差按周期 2048 分组** → 是变换里某一步错了，周期 2048 指出是哪一层。

`slot%2048` 的 spread (0.015) 是 `slot%16` (0.0007) 的 **23 倍**。
误差有明显的 2048 周期结构，不是均匀噪声。

### 2048 是什么

THOR 的块对角 layout：`slot = group*2048 + token*16 + block`。
2048 是一个 group 的大小（128 tokens × 16 blocks）。

在 bootstrap 的 CtS/StC 线性变换中，level budget (3,3) 把 32768 slots
分成 3 层 CoeffsToSlots + 3 层 SlotsToCoeffs。每层的对角线数量取决于
budget 分配。如果某一层的对角线数量和 2048 有关，那就是出错的那一层。

---

## 5. 因果链更新

```
bootstrap 的 CtS/StC 线性变换引入加性底噪（0.01-0.06）
  → 底噪有 2048 周期结构
  → 不是消息驱动的，不是 alignToDiagonals 导致的
  → 是 CtS/StC 某一层的明文乘或 key switch 引入的
  → 10.5 bit 精度 = 底噪 / (q0/Delta) = 0.02 / 32 ≈ 2^-10.5
  → he_inv 的 2e-4 分母被 0.02 底噪完全淹没
  → Goldschmidt 发散
```

---

## 6. 下一步

1. **往 CtS/StC 的明文对角线查**：2048 周期结构指向某一层。检查 level budget (3,3)
   下各层对角线的编码精度——如果对角线本身编码时精度不够（比如在太低的 level 编码），
   乘上去的明文就自带噪声。
2. **尝试不同 level budget**：(4,4) 或 (5,2) 看底噪是否改善。budget 直接决定
   CtS/StC 的近似阶数。
3. **检查 `EvalLinearTransform` 的预计算**：`Bootstrap.cu:213` 的 `sparse_encaps`
   为 false（N/2 == slots），走非 sparse 路径。检查非 sparse 路径的 `EvalLinearTransform`
   对角线编码。

208 passed, 7 skipped（重编译后测试全通过）。
