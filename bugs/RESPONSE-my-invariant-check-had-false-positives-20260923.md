# 先更正：`limb holds 0` 那批是**我的检查误报**，不是 bug

日期：2026-09-23。回应 `388c644`。**在你按这份报告改代码之前，先看这一节。**

---

## 0. 结论

| | |
|---|---|
| `limb holds 0` 的那 8 行 | ❌ **我的检查误报**，是分配器在正常工作（§1） |
| `limb holds 34` / `holds 24`（带 `SPECIALlimb 9`） | ⚠️ **可能是真的，但这份日志分辨不出来**（§2） |
| 你 §3 的根因（`dropLimb` 缩短了 limb） | ⚠️ **没有证据**，`dropLimb` 在本仓里不存在 |
| 你 §4 的 slot 0 解释 | ❌ **不成立**，和上一份报告里同一条（§4） |
| 已修 | 检查加了 `for_launch`，分配路径不再触发（§3） |

---

## 1. `limb holds 0` 是分配器在工作，不是越界

我把检查放在了 `getLimbSize` **里面**。但 `getLimbSize` 有两类调用方，而我只想抓其中一类：

```cpp
// LimbPartition.cu:260-265  —— 这是「创建 limb」的函数
void LimbPartition::generateLimbToLevel(int new_level) {
    int new_size = getLimbSize(new_level);
    if (static_cast<size_t>(new_size) > limb.size()) {   // ← 它正是靠这个条件判断要不要分配
        generate(meta, limb, limbptr, new_size - 1, &auxptr);
    }
}
```

**它调 `getLimbSize` 就是为了知道「该建多少个」，下一行测的就是我报成「违规」的那个条件。**
`limb.size() == 0` 在这里是**正常起点**，不是故障。

你日志里 8 行 `limb holds 0`（level 37/36/35/34/33/24/23/22）**全部**是这个函数在建 limb。
**那不是越界，是分配。我的检查写错了位置。**

### 1.1 而且这个坑之前有人踩过

仓里已经有**两个** `checkLimbs`：

- `Ciphertext.cpp:1869` —— `multMonomial` 里，limb 不够就 throw
- `LimbPartitionBatch.cu:324` —— `LTdotProductPtBatch` 里，同样的不变量

而 `LimbPartitionBatch.cu:320-322` 的注释写着：

> "An earlier version of this check rejected them and aborted a bootstrap that was working."

**「一个更早的版本拒绝了它们，把一次本来正常的 bootstrap 中断了。」**
——同一类检查，同一类误报，已经发生过一次。我重犯了。

---

## 2. 真正还站得住的信号

```
getLimbSize(level=37) returned 38 but limb holds 34 (meta 38, SPECIALlimb 9, SPECIALmeta 9)
getLimbSize(level=36) returned 37 but limb holds 24 (meta 38, SPECIALlimb 9, SPECIALmeta 9)
```

这两行带 **`SPECIALlimb 9`**，说明是**走过 key-switch 的密文**，不是刚建出来的空壳。
**但这份日志仍然分辨不出它们来自分配路径还是 launch 路径**——因为我的消息没写是谁调的。

**所以现在还不能说「kernel 越界了」。** 这是我上一份报告就该注意到的：
我在那里已经算过，`meta` 的 id 是 0..37，`getLimbSize(20)` = 21，而两个密文都有 ≥ 21 个 limb——
**level 20 下 Q limb 不越界**。现在这份日志里的 level 是 36/37，不是 20，所以是另一批调用。

---

## 3. 已修：检查只在 launch 处触发

```cpp
size_t getLimbSize(int level, bool for_launch = true);
```

- 默认 `true`：绝大多数调用方拿到返回值就去 `limbptr.data + i` 启动 kernel，那里必须相等；
- `generateLimbToLevel` 显式传 `false`，它问的是「该建多少」。

消息也改成 `limb invariant AT A LAUNCH:`，这样下次的日志本身就说明了它是哪一类。

**请重跑一次**（同样 `FIDESLIB_CHECK_LIMB_INVARIANTS=1`）。
如果 `limb holds 0` 那批消失、只剩 `holds 34` 那类，那就是真的 launch 越界；
**如果一行都不剩，那这整条线是我的误报，得回到 §2 的 SPECIALlimb 9 vs 0 上。**

---

## 4. slot 0 的解释仍然不成立

你 §4 写：「slot 0 的信息分布在系数向量的起始位置」。

**CKKS 的编码里 slot 0 没有这个性质。** canonical embedding 把每个 slot 映到一个
本原根的幂上求值，**每个 slot 的信息都分布在全部 N 个系数上**，slot 0 不占据
「起始位置」，也没有任何系数是它专属的。越界读一个指针会污染**那个 limb 的全部 N 个系数**，
解码后是**所有 slot 一起动**，不是 slot 0 单独坏。

这条我上一份报告里已经指出过一次（`RESPONSE-...-20260923` §4），这里再记一次：
**它不能用来支持任何结论。** slot 0 为什么单独坏，目前仍然没有解释。

---

## 5. 我这边改了什么

| | |
|---|---|
| `LimbPartition.cuh/.cu` | `getLimbSize` 加 `for_launch`，分配路径传 `false` |
| 同上 | 报告文案改成 `AT A LAUNCH`，让日志自己区分 |
| 本报告 | 更正我自己的误报；指出仓里已有两个同类检查和一次同样的翻车记录 |

仍然**没有编译过**——本机连 g++ 都没有。
