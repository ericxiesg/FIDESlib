# 两次修复尝试均失败：freeSpecialLimbs 和 zero_out 都没修 slot 0

日期：2026-09-23。针对 `6c45a36`。

---

## 0. 结论

| 尝试 | 改动 | 结果 |
|---|---|---|
| 1. bootstrap 出口 freeSpecialLimbs | `Bootstrap.cu:362` 加 `ctxt.c0.freeSpecialLimbs(); ctxt.c1.freeSpecialLimbs();` | ❌ slot 0 仍 12854x, worst at 0/9891/20170 |
| 2. ModRaise 里 zero_out=true | `generateSpecialLimbs(false, true)` → `generateSpecialLimbs(true, true)` | ❌ slot 0 仍 12622x, worst at 0/9891/20170 |

**SPECIALlimb 的陈旧数据假设被排除。** 清零和释放都没改变 slot 0 的错误。

---

## 1. 尝试 1：freeSpecialLimbs at bootstrap exit

在 `Bootstrap.cu` 的完整 bootstrap 退出点（`stopAfterStage == -1` 走到底）加：
```cpp
ctxt.c0.freeSpecialLimbs();
ctxt.c1.freeSpecialLimbs();
ctxt.slots = old_slots;
```

### 结果

```
raw (no rescale loop):      slot 0 / p50 = 12854.6x   error 1.43322   worst 0, 9891, 1, 19782, 20170
corrected (current path):   slot 0 / p50 = 12869.3x   error 1.41646   worst 0, 9891, 1, 20170, 19782
```

**和 baseline 完全一样。** 实验 5 确认 SPECIALlimb 仍然是 9（freeSpecialLimbs 执行了但被后续操作重新生成了，或者没传到正确的 partition）。

### 分析

`freeSpecialLimbs` 清空了 `SPECIALlimb` vector 和 `bufferSPECIAL`。但：
- Python 侧的 `engine.bootstrap` 在 `EvalBootstrap` 返回后做了 rescale loop
- rescale 本身不重新生成 special limbs
- 但 `_times`（multiply → relinearize → rescale）中的 `Ciphertext::mult` 调用了 `extend()` → `generateSpecialLimbs`

**所以 free 在 bootstrap 出口没用——multiply 会重新生成。**

---

## 2. 尝试 2：zero_out=true in ModRaise

把 `Bootstrap.cu:664` 和 `:717` 的 `generateSpecialLimbs(false, true)` 改成 `generateSpecialLimbs(true, true)`——强制清零而不是复用陈旧数据。

### 结果

```
raw (no rescale loop):      slot 0 / p50 = 12622.9x   error 1.39206   worst 0, 9891, 1, 20170, 19782
corrected (current path):   slot 0 / p50 = 13424.7x   error 1.45084   worst 0, 9891, 1, 20170, 19782
```

**仍然一样。** 清零 special limbs 的数据没有改变 slot 0 的错误。

### 分析

**SPECIALlimb 的陈旧数据不是 slot 0 损坏的原因。** 即使在 ModRaise 入口强制清零，bug 仍然出现。

---

## 3. 已排除

| 假设 | 排除方式 |
|---|---|
| SPECIALlimb 残留数据污染 | 尝试 1（free）+ 尝试 2（zero_out）都没修 |
| bootstrap 出口需要清理 | free 之后 multiply 会重新生成 |

## 4. 仍然站得住

- bootstrap 产出密文 `limb=34, SPECIALlimb=9`，dropped `limb=38, SPECIALlimb=0`
- `limb` 差 4 已被 `dropToLevel` 解释（`if (0 && ...)` 关掉了丢弃逻辑）
- `SPECIALlimb` 差 9 是 bootstrap 路径和 fresh encrypt 路径的正常差异
- **但两者都不是 slot 0 bug 的原因**

## 5. 下一步

slot 0 bug 不在 limb 表大小、不在 special limbs 残留数据。差异在 **limb 的内容**——bootstrap 产出的密文在 slot 0 对应的系数上和 fresh encrypt 不同，而这个差异在低 level multiply 时被放大。

需要对比 bootstrap 后和 fresh encrypt 后的 **limb 数据内容**（不只是表大小），特别是 slot 0 对应的系数。这可能需要在 C++ 侧加 per-coefficient 的打印。
