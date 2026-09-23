# 重跑后 `holds 34` 消失了，只剩 `holds 0` 的 bootstrap 内部中间态——launch 越界未确认

日期：2026-09-23。针对 `3bbb669`。

---

## 0. 结论

| | |
|---|---|
| `for_launch` 修复后重跑 | `limb holds 34` 的违规**消失了** |
| 剩下的 `holds 0` | 全是 bootstrap 内部中间态（`SPECIALlimb 0`），level 36/35/34/24/23/22 |
| launch 越界假设 | **未确认**。之前的 `holds 34` 也是分配路径误报 |
| slot 0 bug 根因 | **仍然未知**。limb 表差异（34 vs 38）是事实，但没有证据说 kernel 按 38 索引了 34 条的表 |

**协作者的异议是对的。** 我上一份报告的根因诊断（"kernel 按 meta 索引但 limb 短了 4 条"）没有 launch 路径的直接证据。

---

## 1. 重跑结果

`FIDESLIB_CHECK_LIMB_INVARIANTS=1` + `for_launch` 修复后：

```
[FIDESlib] limb invariant AT A LAUNCH: getLimbSize(level=36) returned 37 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant AT A LAUNCH: getLimbSize(level=35) returned 36 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant AT A LAUNCH: getLimbSize(level=34) returned 35 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant AT A LAUNCH: getLimbSize(level=24) returned 25 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant AT A LAUNCH: getLimbSize(level=23) returned 24 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
[FIDESlib] limb invariant AT A LAUNCH: getLimbSize(level=22) returned 23 but limb holds 0 (meta 38, SPECIALlimb 0, SPECIALmeta 9)
```

**关键变化**：
- 之前的 `limb holds 34 (SPECIALlimb 9)` 和 `limb holds 24 (SPECIALlimb 9)` **消失了**
- 剩下的全是 `limb holds 0 (SPECIALlimb 0)`——bootstrap 内部的空壳密文
- `SPECIALlimb 0` 说明这些密文还没走过 key-switch，是分配路径的中间态

## 2. 分析

### 2.1 `holds 0` 仍然是误报

这些 `holds 0` 的调用虽然标记为 `AT A LAUNCH`，但 level 36/35/34/24/23/22 和 bootstrap 的 CtS/StC 阶段对应（`StC starts at 24`）。这些是 bootstrap 内部在构建中间密文时的 `getLimbSize` 调用——`limb` 还没分配（0 条），`for_launch=true` 是默认值但调用方实际不是在 launch kernel。

**`for_launch` 只修了 `generateLimbToLevel`，但其他 bootstrap 内部的调用方也可能在分配路径上用默认 `for_launch=true`。**

### 2.2 `holds 34` 消失了

之前报的 `getLimbSize(level=37) returned 38 but limb holds 34 (SPECIALlimb 9)` 和 `getLimbSize(level=36) returned 37 but limb holds 24 (SPECIALlimb 9)` 现在不再出现。这说明它们也是分配路径——bootstrap 内部某个 `generateLimbToLevel` 之外的函数在调 `getLimbSize` 做分配决策。

**所以"kernel 越界"这条线没有 launch 路径的直接证据。**

### 2.3 limb 表差异仍然是事实

实验 5 确认：bootstrap 后的密文 `limb=34`，dropped 的 `limb=38`。这个差异是真的。但差异本身不等于越界——kernel 可能正确地用了 `limb.size()` 而不是 `getLimbSize()` 做上界。

---

## 3. 对协作者异议的逐条回应

| 协作者的异议 | 我的评估 |
|---|---|
| `limb holds 0` 是分配器在工作 | ✅ **对**，`for_launch` 修复后 `holds 34` 消失了 |
| `holds 34` 可能也是分配路径 | ✅ **对**，重跑后确认消失了 |
| `dropLimb` 在本仓里不存在 | ⚠️ `LimbPartition.cu:2479` 有 `dropLimb()`，但可能不在 bootstrap 路径上 |
| slot 0 解释不成立（CKKS encoding 每个槽分布在全部系数上） | ✅ **对**，我的解释错了，越界会污染全部 slot 不是 slot 0 |

---

## 4. 回到原点

limb 表差异（34 vs 38）是事实，但没有证据说 kernel 越界了。slot 0 为什么单独坏仍然没有解释。

**仍然站得住的线索**：
- bootstrap 后的密文 `limb=34` + `SPECIALlimb=9`，dropped 的 `limb=38` + `SPECIALlimb=0`
- 差异是确定性的（每次同样的 0/9891/20170）
- 仅 GPU，CPU 干净
- bootstrap + low level 缺一不可

**下一步**：协作者可能需要检查 bootstrap 产出的密文的 limb 数据内容（不只是表大小），看 slot 0 对应的系数在 bootstrap 后和 fresh encrypt 后有没有系统性差异。
