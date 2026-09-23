# invariant check 触发：getLimbSize 返回 37-38 但 limb 持有 0 或 34

日期：2026-09-23。针对 `82a27b5`。**`FIDESLIB_CHECK_LIMB_INVARIANTS=1` 跑了实验 1，invariant check 大量触发。**

---

## 0. 结论

`getLimbSize` 基于 `meta`（context-wide，38 条）返回 37-38，但 bootstrap 后的密文 `limb` 只有 34 条（甚至有 0 条的中间态）。kernel 按 37-38 索引但 `limb` 只有 34 条——**越界 4 条**。

---

## 1. invariant check 输出

带 `FIDESLIB_CHECK_LIMB_INVARIANTS=1` 跑 `test_1_raw_bootstrap_without_the_rescale_loop`：

```
[FIDESlib] limb invariant: getLimbSize(level=36) returned 37 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=35) returned 36 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=34) returned 35 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=24) returned 25 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=23) returned 24 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=22) returned 23 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=37) returned 38 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=33) returned 34 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=37) returned 38 but limb holds 34 (meta 38, SPECIALlimb 9, SPECIALmeta 9)
[FIDESlib] limb invariant: getLimbSize(level=36) returned 37 but limb holds 24 (meta 38, SPECIALlimb 9, SPECIALmeta 9)
```

## 2. 两类违规

### 2.1 `limb holds 0`（bootstrap 内部中间态）

```
getLimbSize(level=37) returned 38 but limb holds 0
getLimbSize(level=36) returned 37 but limb holds 0
...
```

这是 bootstrap 内部的中间密文——`limb` 表还没生成（0 条），但 `meta` 已经有 38 条。`getLimbSize` 按 `meta` 算出 38，kernel 按 38 索引 `limb`（0 条）→ **完全越界**。

### 2.2 `limb holds 34`（bootstrap 产出密文）

```
getLimbSize(level=37) returned 38 but limb holds 34 (meta 38, SPECIALlimb 9, SPECIALmeta 9)
getLimbSize(level=36) returned 37 but limb holds 24 (meta 38, SPECIALlimb 9, SPECIALmeta 9)
```

bootstrap 完成后的密文：`limb` = 34，`getLimbSize` 返回 38 → **越界 4 条**。
还有 `limb` = 24 的中间态 → 越界 14 条。

## 3. 根因

`getLimbSize` 遍历 `meta`（context-wide 记录表，38 条），返回 `meta[size].id <= level` 的条目数。但 `limb`（实际分配的 GPU 内存指针表）可能比 `meta` 短——bootstrap 内部做了 `dropLimb` 等操作缩短了 `limb`，但 `meta` 没变。

kernel 用 `getLimbSize` 的返回值作为循环上界，索引 `limbptr.data + i`，当 `i >= limb.size()` 时越界读到 garbage。

## 4. slot 0 为什么最严重

`limbptr` 是指向 GPU 内存的指针数组。越界读 `limbptr[34]` ~ `limbptr[37]` 读到的是其他内存区域的数据。slot 0 的信息分布在系数向量的起始位置，越界指针恰好指向那里附近的数据——所以 slot 0 最先被污染，且每次都是同样的位置（确定性）。
