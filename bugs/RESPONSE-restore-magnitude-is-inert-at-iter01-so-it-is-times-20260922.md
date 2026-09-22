# `_restore_magnitude` 在 iter01 **什么都没做**——所以坏的是 `_times`

日期：2026-09-22。回应 `ef53ec2`。**本机参考跑出来了，它直接排除了你的结论。**

---

## 0. 结论

| | |
|---|---|
| `_restore_magnitude` 是 iter01 的坏步骤吗 | ❌ **不可能**。它在 iter01 **完全是空操作**（§1） |
| 怎么证明的 | `delta_headroom_bits=8`，iter01 时 `b.delta=0.274`，`int(1/0.274/256)=0` → 不加倍、factor=1、两个分支都不进 |
| 那坏在哪 | **`_times(b.ciphertext, correction)`**，而同一个 `correction` 上一行刚喂给 `a`，`a` 是好的（§2） |
| 下一步 | **`THORFHE_SWAP_TIMES=1`**：把两次 `_times` 换序。一次跑分辨"别名"还是"操作数"（§3） |

---

## 1. 本机参考：iter01 的 `_restore_magnitude` 是空操作

加了 `inv_iter{n}_{a,b}_times`（`_times` 之后、`_restore_magnitude` 之前）后，本机：

```
07.inv_iter01_a_times   max +0.9676   worst at 18868(%4),6580(%4),31156(%4)
07.inv_iter01_a         max +0.9676   worst at 18868(%4),6580(%4),31156(%4)   ← 逐位相同
07.inv_iter01_b_times   max +0.274    worst at 31370(%10),29322(%10),27274(%10)
07.inv_iter01_b         max +0.274    worst at 31370(%10),29322(%10),27274(%10) ← 逐位相同
```

**前后完全一样。** 不是"几乎一样"，是同一个数。

### 1.1 为什么

```python
headroom = 2 ** self.delta_headroom_bits        # delta_headroom_bits = 8  ->  256
if int(1 / b.delta / headroom) > 1:             # 加倍分支
factor = max(int(1 / b.delta / headroom), 1)    # 整数乘分支
```

逐轮算 `b.delta`（这是纯 Python 的记账，两边**完全相同**）：

| iter | `b.delta` | `int(1/delta/256)` | `_restore_magnitude` |
|---|---|---|---|
| **01** | **0.27399** | **0** | **空操作** |
| **02** | **0.018767** | **0** | **空操作** |
| 03 | 8.8052e-05 | 44 | 加倍 + ×44 |
| 04 | 1.501e-05 | 260 | 加倍 + ×260 |
| 05 | 1.523e-05 | 256 | 加倍 + ×256 |

> **iter01 和 iter02，`_restore_magnitude` 一条指令都不执行。**
> **`delta` 的记账在设备上是同一份 Python 代码，所以设备上也一样。**

**于是你 §3/§4 的结论被排除了**：不能是 `_restore_magnitude` 干的，那一轮它不存在。
你 §4 建议"在 `_restore_magnitude` 内部再二分"——那会花在一个空操作上。

（顺带：这也再次说明 §1 更正的那件事——`iter01_a` 和 `iter01_b` 两个探针之间
本来就没有操作，现在连 `_restore_magnitude` 本身在那一轮都不做事。）

---

## 2. 所以坏在 `_times`，而且范围很窄

一轮迭代在 iter01 实际只有三件事：

```python
correction = subtract(2/k * b.delta, b.ciphertext)   # 标量 − 密文
a_new = _times(a.ciphertext, correction)             # ← 干净
b_new = _times(b.ciphertext, correction)             # ← 坏
# _restore_magnitude：空操作
```

把设备的三个观测放在一起：

| 观测 | 设备 | 含义 |
|---|---|---|
| `inv_input_lift3` max 在 `%3`，slot 0 不在 worst-3 | ✅ 健康 | **`b₀` 在 slot 0 是好的** |
| `iter01_a` max 在 `%4`，slot 0 不在 worst-3 | ✅ 健康 | `a = ones`，slot 0 处值为 1，所以 `a_new = 1 × correction` → **`correction` 在 slot 0 是好的** |
| `iter01_b` max **在 slot 0**，1.541 | ❌ 坏 | **`b₀ × correction` 在 slot 0 错了** |

> **两个操作数在 slot 0 都是好的，乘积不是。**
> **而且乘积超出了映射的值域（见 `957efaf`：判别式 −18.5，没有实数 b₀ 能产生它）。**

`_times` = `rescale(relinearize(multiply(x, y)))`。**范围缩到这三个算子。**

---

## 3. 下一步：换序，一次跑分辨别名

两次 `_times` **共用右操作数 `correction`，此外没有任何关系**，
所以先算哪个在算术上完全等价。而设备的症状是：**第一次干净，第二次坏。**

这正是"第一次调用破坏了共享的 `correction`，第二次拿到的是被破坏的版本"的形状——
**也就是别名/原地修改**。

已推 `THORFHE_SWAP_TIMES`：

```bash
THORFHE_SWAP_TIMES=1 THORFHE_DEBUG=1 ... --inverse-lift 3 --check-ranges warn
```

它把顺序换成先 `b` 后 `a`。判读只有两种，没有中间地带：

| `iter01` 的坏槽跑到哪 | 结论 |
|---|---|
| **跟着第二次调用跑到 `a`** | **是别名。** `_times` 的第一次调用在原地改了 `correction` |
| **仍然在 `b`** | **不是别名**，是操作数决定的——那就去查 `multiply`/`relinearize`/`rescale` 对这个特定值 |

**换序在本机是严格中性的**，我加了测试钉住（`test_swapping_the_two_times_calls_changes_nothing`，
直接比 `he_inv` 的返回值，`rel=1e-12`）。所以设备上如果坏槽动了，那是别名，不是我的改动。

---

## 4. 请跑

一次跑，两个环境变量，贴 `iter01` 和 `iter02` 的四行
（`_a_times` / `_b_times` / `_a` / `_b`）：

```bash
THORFHE_SWAP_TIMES=1 THORFHE_DEBUG=1 python3 -u -m thorfhe.bench fhe \
    --engine fideslib --device cuda:0 ... --inverse-lift 3 --check-ranges warn --per-stage
```

顺便把不带 `THORFHE_SWAP_TIMES` 的同样四行也贴一次（`957efaf` 之后你还没跑过带
`_times` 探针的版本），两边一对就出来了。

本机健康参考（`--compact`，同参数）：

```
07.inv_iter01_a_times   max +0.9676   worst at 18868(%4),6580(%4),31156(%4)
07.inv_iter01_b_times   max +0.274    worst at 31370(%10),29322(%10),27274(%10)
07.inv_iter02_a_times   max +0.2362   worst at 6580(%4),27060(%4),22964(%4)
07.inv_iter02_b_times   max +0.02574  worst at 6709(%5),12853(%5),31285(%5)
```

---

## 5. 我这边改了什么

| 改动 | |
|---|---|
| `numeric.py` | `THORFHE_SWAP_TIMES`：换 `_times` 的调用顺序，诊断用，不是调参开关 |
| `test_division_padding_floor.py` | 钉住换序在本机严格中性（`rel=1e-12`） |
| 本报告 | `_restore_magnitude` 在 iter01/02 是空操作的逐轮 delta 表；把范围缩到 `_times` |
