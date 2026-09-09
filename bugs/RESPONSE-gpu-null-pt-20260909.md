# 回复 GPU-null-pt-partition-20260909

日期：2026-09-09。本机没有 GPU，**C++ 改动一行都没编译过**。

**两个都是我的错，都已修。** 而且第一个不是根因——是我加的检查误报了。

---

## 一、编译错误：我的

`KeySwitchingKey` 有 `CudaNvtxRange my_range` 成员，赋值被删除，所以

```cpp
existing->second = std::move(ksk);      // ← 编译不过
```

**你的 erase + emplace 修法是对的**，我按同样的方式改进仓库了，你不用再本地打补丁、
也不会被 `git reset --hard` 冲掉。抱歉——这一行我本该在写的时候就注意到。

## 二、`null pt partition`：不是 bug，是我的检查写错了

`pt` 数组里出现 nullptr 是**合法且预期的**。`LinearTransform.cu` 的 `DotProductPtInternal`：

```cpp
for (auto& i : b) {
    if (i) pts.push_back(&i->c0);
    else   pts.push_back(nullptr);      // ← 故意的
}
```

主机侧构造设备指针表时也一路带着它：

```cpp
pt[...] ? pt[...]->limbptr.data : nullptr;
```

而 kernel 里有两处 `if (pt_partition != nullptr)`，**遇到空的就跳过这一项**。
也就是说「某条对角线没有对应明文」是这个线性变换的正常情形，index 31 为空完全正常。

**是我上一轮把 nullptr 当成错误了**，于是把一个本来能跑的 bootstrap 直接 abort 掉——
`test_stage4_bootstrap` 之前是过的，是我弄崩的。已改：**空指针跳过，不再报错**。

## 三、但那个检查的另一半要留着，而且仍然是最可疑的一处

注意 kernel 是这样取值的：

```cpp
if (pt_partition != nullptr) {
    pt[0] = ((uint64_t*)pt_partition[blockIdx.y])[idx];   // ← 只校验了外层指针
}
```

它只检查 `pt_partition` 本身非空，**没有检查 `blockIdx.y` 是否在这个 partition 的 limb 范围内**。
而 `grid.y = limbsize` 是从 `out[0]` 的 level 算出来的。所以：

> partition **存在**、但持有的 limb 数少于 `out[0]` level 所要求的 → `pt_partition[blockIdx.y]`
> 越界 → 拿到垃圾指针 → 非法访存。

这仍然是最符合现象的解释（自举明文按自己的 level 预计算，和密文当前 level 不是一回事），
所以 **limb 数检查保留**，只是不再把 nullptr 当错误：

```
FIDESlib: LTdotProductPtBatch would read 29 limbs from pt[7], which holds 24.
The operands are at different levels.
```

## 四、现在的状态，说清楚

| | |
|---|---|
| `null pt` | **不是 bug**，是我误报，已修 |
| 编译错误 | 我的，已修 |
| depth=34 的那次非法访存 | **仍未解释**。它出现在 `3f72760`，早于我改自举密钥计划，所以也不是那个引起的 |

也就是说这一轮**净收益是排除了一个假设**，外加把我自己制造的两个问题清掉了。
下一次跑如果那句 limb-count 异常出现，就直接拿到根因；如果没出现而是又回到裸的
illegal address，那说明越界不在 limb 数上，得换个方向——那时我会去查
`grid.y` 之外的索引（`blockIdx.z` 的 `num_parallel_parts` 分块，和 `offset_*` 那组偏移）。

---

## 五、下一次运行建议

1. **拉这版跑 `pytest`**：`test_stage4_bootstrap` 应该恢复通过（我的误报去掉了）。
   这一步先确认我没再引入新的东西。
2. **再跑 depth=34 那组 THOR 一层**。两种结果都有用：
   - 抛出 `LTdotProductPtBatch would read N limbs from X[k]` → 拿到根因，把那行发我；
   - 又是裸的 illegal address → 说明不是 limb 数，我换方向查索引算术。
3. 顺带确认 `--allow-key-grow` 下 "grown at runtime" 是 0（上一轮 `AddRotationKeys`
   「有键就跳过」的修复应该让它变成 0 了）。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CKKS/Context.cu` | `AddRotationKey` 改用 erase + emplace（`KeySwitchingKey` 不可赋值）**未编译** |
| `src/CKKS/LimbPartitionBatch.cu` | `LTdotProductPtBatch` 的检查：nullptr 跳过而非报错，limb 数检查保留 **未编译** |
