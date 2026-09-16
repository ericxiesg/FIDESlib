# 三个数据点：slope ≈ 1.0 bit/bit（同 q0/Delta 下），22 bit 需要 fmb ≈ 66

日期：2026-09-16。针对 `a3d5909`。

---

## 三个数据点

为了公平比较，需要保持 q0/Delta 一致（否则 sine 近似误差混入）。
sb=55/fmb=55 的 q0/Delta=1，测试值 2.0/-3.0 超界，不可用。
改用 fmb=60 保持 q0/Delta=32。

```
sb=50 fmb=55 q0/Delta=32: max_error=1.57e-02  precision=11.0 bits
sb=52 fmb=55 q0/Delta=8:  max_error=6.66e-04  precision=13.6 bits
sb=55 fmb=60 q0/Delta=32: max_error=4.44e-04  precision=16.1 bits
```

## 斜率分析

### 同 q0/Delta=32 的比较（干净）

```
sb=50 → sb=55:  Delta 2^50 → 2^55 (5 bits)
  precision: 11.0 → 16.1 (5.1 bits)
  slope = 5.1 / 5 = 1.02 bits/bit
```

**slope ≈ 1.0，误差 ∝ 1/Delta 确认。** 每多 1 bit Delta，精度多 1 bit。

### sb=52/fmb=55（q0/Delta=8）这个点

```
sb=50→sb=52 (同 fmb=55, q0/Delta 从 32→8):
  Delta 2^50 → 2^52 (2 bits)
  precision: 11.0 → 13.6 (2.6 bits)
  slope = 2.6 / 2 = 1.30 bits/bit
```

这个 slope (1.30) 比干净比较的 1.02 陡。原因是 q0/Delta 从 32 降到 8，
sine 近似误差也降低了（消息占界比例从 3/32=9.4% 到 3/8=37.5%——
但 sine 误差不是线性的，在更小的界上正弦近似更好）。

**协作者之前从两点算出的 1.40 bits/bit 包含了 q0/Delta 变化的混合效应。**
干净的 slope 是 1.02。

### 常数 C 的验证

```
sb=50: C = error × Delta = 1.57e-2 × 2^50 = 1.77e13 = 2^43.99
sb=55: C = error × Delta = 4.44e-4 × 2^55 = 1.60e13 = 2^43.87
```

**同 q0/Delta=32 下，C ≈ 2^44，两个点一致。** 误差 = C/Delta 确认。

sb=52 的 C = 6.66e-4 × 2^52 = 3.00e12 = 2^41.47 — 比另外两点小 2^2.5。
这是因为 q0/Delta=8 时 sine 近似更精确，常数 C 本身变小了。

---

## 外推到 22 bit

```
precision = log2(q0) - log2(C) = log2(q0) - 44

For 22 bits: log2(q0) = 22 + 44 = 66 → q0 = 2^66 → first_mod_bits = 66
```

**22 bit 需要 first_mod_bits ≈ 66，超过 64 位 limb 上限。**

即使 q0/Delta=2（EasyFHE）：
```
precision = log2(q0) - 44 = log2(2 × Delta) - 44
For 22 bits: Delta = 2^(22+44-1) = 2^65 → scaling_bits = 65
```

仍然超过 64 位。**在 64 位 RNS 下无法达到 22 bit bootstrap 精度。**

### 但 sb=55 已经 16 bit

```
sb=55 fmb=60: 16.1 bits, error = 4.44e-4
```

he_inv 的分母下界是 4.88e-4。**error (4.44e-4) < inv_epsilon (4.88e-4)**。
在 sb=55 下，bootstrap 误差已经小于 he_inv 的下界——**he_inv 可能刚好能用。**

但要考虑分母 p50 = 7.03e-5（当前标定），远低于 4.88e-4。标定修好后分母回到
~4.88e-4，误差 4.44e-4 是分母的 91%。**勉强够，但余量极薄。**

### sb=58 外推

```
sb=58 fmb=60: q0/Delta=4, precision ≈ 16.1 + 3×1.02 = 19.2 bits
  error ≈ 4.44e-4 / 8 = 5.55e-5
  vs inv_epsilon = 4.88e-4 → error is 11% of inv_epsilon → good
  but q0/Delta=4, usable bound |v| < ~1.3 → stage 15 更糟
```

---

## 对协作者 §1 更正的更正

协作者指出我上轮的 C/Delta 计算有误（差 7x）。用三个点重新算：

```
sb=50/fmb=55: C = 1.57e-2 × 2^50 = 2^44.0
sb=52/fmb=55: C = 6.66e-4 × 2^52 = 2^41.5  ← q0/Delta=8, C 更小
sb=55/fmb=60: C = 4.44e-4 × 2^55 = 2^43.9
```

sb=50 和 sb=55（同 q0/Delta=32）给出 C ≈ 2^44，一致。
sb=52（q0/Delta=8）给出 C ≈ 2^41.5，更小——**C 本身依赖 q0/Delta。**
更窄的界让 sine 更精确，C 更小。

所以误差模型更准确地是：**error = f(q0/Delta) / Delta**，其中 f 随 q0/Delta 变窄而减小。

---

## 对两 bug 冲突的确认

协作者 §2 的分析正确：

```
sb=50: q0/Delta=32, bound ~10, 0.2% 坐标越界
sb=58: q0/Delta=4,  bound ~1.3, ~34% 坐标越界
```

**抬 Delta 修 Bug 1 会恶化 Bug 2。** 协作者 §3 的提议（把 1/128 折进 stage 11/14 的
明文权重，调整 stage 16 的 min_var/max_var/var_e）是正确的出路——不花 level，
利用 LayerNorm 的尺度不变性。

---

## 建议的下一步

1. **实现 stage 15 缩放**（协作者 §3 提议）：1/128 折进 gamma/beta 和 output.dense
2. **在 ClearEngine 上验证** stage 16 输出不变（尺度不变性是精确的）
3. **注意 `halve` 标志**（§4）：从绝对值判断改为显式参数
4. **迁移到 sb=55/fmb=60**：16 bit 精度，error=4.44e-4 ≈ inv_epsilon，可能刚好够用
5. **如果 16 bit 不够**：sb=58/fmb=60 给 19 bit，但需要 stage 15 缩放先到位

---

## 测试

218 passed, 13 skipped。
