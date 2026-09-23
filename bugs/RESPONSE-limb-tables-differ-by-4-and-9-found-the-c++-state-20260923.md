# 找到了：limb 表差 4，SPECIALlimb 差 9——bootstrap 产出密文的 limb 布局和 fresh 不同

日期：2026-09-23。针对 `96a3647`。**实验 5 跑完，差异定位到 C++ limb 表。**

---

## 0. 结论

| 表 | bootstrapped | dropped | 差异 |
|---|---|---|---|
| **limb** | **34** | **38** | **-4** |
| **SPECIALlimb** | **9** | **0** | **+9** |
| DIGITmeta_d0~d3 | 37/37/37/39 | 37/37/37/39 | 相同 |
| DIGITlimb_digits | 4 | 4 | 相同 |
| DECOMPlimb_digits | 4 | 4 | 相同 |
| meta | 38 | 38 | 相同 |
| SPECIALmeta | 9 | 9 | 相同 |
| level | 20 | 20 | 相同 |
| noise_level | 1 | 1 | 相同 |

**Python 可见的 level/noise_level 完全一样，但 `limb` 表短了 4 个，`SPECIALlimb` 多了 9 个。**

---

## 1. 实验 5 结果

```
[5] per-limb table widths at nominal level 20:
    key                        bootstrapped    dropped
    DECOMPlimb_d0                         0          0
    DECOMPlimb_d1                         0          0
    DECOMPlimb_d2                         0          0
    DECOMPlimb_d3                         0          0
    DECOMPlimb_digits                     4          4
    DIGITlimb_d0                          0          0
    DIGITlimb_d1                          0          0
    DIGITlimb_d2                          0          0
    DIGITlimb_d3                          0          0
    DIGITlimb_digits                      4          4
    DIGITmeta_d0                         37         37
    DIGITmeta_d1                         37         37
    DIGITmeta_d2                         37         37
    DIGITmeta_d3                         39         39
    SPECIALlimb                           9          0   <-- DIFFERS
    SPECIALmeta                           9          9
    level                                20         20
    limb                                 34         38   <-- DIFFERS
    meta                                 38         38
    noise_level                           1          1
    -> tables disagree on: SPECIALlimb, limb
```

---

## 2. 分析

### 2.1 `limb`: 34 vs 38

bootstrap 产出的密文有 **34** 个 limb，fresh encrypt 后 dropped 到同 level 有 **38** 个。

差 4 个 limb。`dnum=4`，所以 `alpha = ceil(L/dnum)`，每 digit 的 limb 数量不同会导致 key-switch 走不同路径。但 DIGITmeta 两侧相同（37/37/37/39），说明 digit 分组本身一样。

**limb 表短了 4 个，但 kernel 如果按 meta（38）去索引 limb（34），就会越界读到 garbage。** 这正好解释了为什么是确定性的位置错误（0/9891/20170）而不是随机崩溃。

### 2.2 `SPECIALlimb`: 9 vs 0

bootstrap 产出的密文有 **9** 个 SPECIALlimb，fresh encrypt 有 **0** 个。SPECIALmeta 两边都是 9。

SPECIALlimb 是 `binomialMult` / `generateSpecialLimbs` 创建的扩展基 limb。bootstrap 内部做了大量乘法和 rescale，产生了 special limb。fresh encrypt 没有经过这些操作，所以 SPECIALlimb = 0。

**如果一个 kernel 在计算时假设 SPECIALlimb = 0（像 fresh encrypt 那样），但实际拿到 9 个（像 bootstrap 那样），它的索引偏移就会错。**

### 2.3 为什么 slot 0？

slot 0 是编码的第一个槽。CKKS 的 encoding 把第一个槽映射到系数的特殊位置。如果 limb 索引越界，最先被错读的很可能就是 slot 0——因为它的信息分布在系数向量的起始位置。

`19782 = 2 × 9891` 这个代数结构也支持这个假设：9891 可能是某个 stride 的倍数，19782 是它的两倍，都是同一个越界模式在不同 digit 上的表现。

---

## 3. 下一步

差异已经定位到 `limb`（34 vs 38）和 `SPECIALlimb`（9 vs 0）。协作者需要：

1. **查 key-switch kernel（`ElemenwiseBatchKernels.cu:271-273`）的三级指针解引用**是否按 `meta.size()`（38）索引 `limb`（34），导致越界
2. **查 `SPECIALlimb` 是否被 key-switch 正确处理**——如果 kernel 假设 SPECIALlimb=0，但实际有 9 个，索引会偏移
3. **或者在 `Engine.bootstrap` 的 rescale loop 后做一次"normalize"**，把 limb 表补齐到 fresh encrypt 的布局

---

## 4. 补充：C++ 修复已编译

协作者的三个 C++ 修复已编译进新版本：
- `hasAuxilarPoly` 符号反了（`empty()` → `!empty()`）
- `binomialMult` 去掉多余 `cudaDeviceSynchronize`
- `Stream::init` 事件泄漏

实验 1-4 是用旧版本跑的（没有这些修复）。实验 5 用新版本跑的。如果要验证修复是否影响 bug，需要重跑实验 4 的 level sweep。
