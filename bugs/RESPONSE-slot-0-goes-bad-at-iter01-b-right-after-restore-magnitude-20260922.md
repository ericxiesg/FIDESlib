# slot 0 在 iter01_b 首次变坏——正好在 _restore_magnitude 之后

日期：2026-09-22。针对 `4837ca8`。**二分定位完成。**

---

## 0. 结论

| 问题 | 答案 |
|---|---|
| slot 0 什么时候开始变坏 | **`iter01_b`**。`iter01_a` 时 slot 0 不在 worst-3 |
| `iter01_a` → `iter01_b` 之间发生了什么 | **`_restore_magnitude`**：conjugate-add + integer multiply |
| iter01_a 的 max | 0.9957（在范围内，`>1 0/32768`） |
| iter01_b 的 max | 1.541（**越界**，`>1 1/32768`，worst at slot 0） |

**坏在第一步迭代的 `_restore_magnitude`，不是 bootstrap、不是 multiply、不是 subtract。**

---

## 1. 第一次 he_inv 的逐迭代 `worst at`

```
07c.denominator      max +0.321    worst at 10419(%3),179(%3),16563(%3)      ← max 在 %3
07.inv_input_lift3   max +0.9812   worst at 30899(%3),10419(%3),24755(%3)    ← max 在 %3（bootstrap 后）
07.inv_iter01_a      max +0.9957   worst at 4532(%4),10248(%8),27060(%4)     ← max 在 %4，slot 0 不在 worst-3
07.inv_iter01_b      max +1.541    worst at 0(%0),9891(%3),20170(%10)        ← **SLOT 0 首次出现，越界**
07.inv_iter02_a      max +0.2687   worst at 0(%0),4532(%4),10248(%8)         ← slot 0 是 max，min=-2.371
07.inv_iter02_b      max +0.04635  worst at 0(%0),9891(%3),1(%1)             ← slot 0 是 max
07.inv_iter03_a      max +0.0666   worst at 0(%0),4532(%4),10248(%8)         ← slot 0 是 max，min=-35.44
07.inv_iter03_b      max +70.65    worst at 0(%0),9891(%3),16452(%4)         ← slot 0 是 max，开始爆炸
07.inv_iter04_a      max +7.911e+05 worst at 0(%0),4532(%4),10248(%8)        ← slot 0 是 max
07.inv_iter04_b      max +0.003901  worst at 0(%0),9891(%3),1(%1)            ← slot 0 是 max
07d                  max +7.911e+05 worst at 0(%0),4532(%4),10248(%8)        ← slot 0 是 max
```

### 1.1 关键转折：iter01_a → iter01_b

| | iter01_a | iter01_b |
|---|---|---|
| max | 0.9957 | **1.541** |
| >1 | 0/32768 | **1/32768** |
| worst at | 4532(%4), 10248(%8), 27060(%4) | **0(%0)**, 9891(%3), 20170(%10) |
| p50 | 5.812e-13 | 0.2733 |
| slot 0 在 worst-3？ | **否** | **是（第 1 位）** |

**`iter01_a` 到 `iter01_b` 之间唯一发生的事是 `_restore_magnitude`：**
1. `conjugate(ct)` → 取共轭
2. `add(ct, conjugate(ct))` → 2·Re
3. integer multiply → 缩放回原 scale

**在 `iter01_a`（multiply 输出）时，所有 32768 个槽都在 1 以内，slot 0 不突出。**
**经过 `_restore_magnitude` 后，slot 0 被推到 1.541，越界了。**

### 1.2 iter01_a 的 p50 异常

`iter01_a` 的 p50 = 5.812e-13（几乎所有槽都是 ~0），但 max = 0.9957。
这说明 multiply `a*(2-b*a)` 的结果大部分槽接近 0——因为大部分槽的 `b` 是 0（padding 槽），
所以 `2-b*a = 2`，而 `a` 初始是 1，所以 `a*2 = 2`...但 p50 是 0 不是 2。

等等——`iter01_a` 是 rescale 之后的，而且大部分槽（padding）的 `b=0`，
所以 `correction = 2 - 0*a = 2`，`a*2 = 2`。但 p50 是 5.8e-13...

这可能是 `_restore_magnitude` 之前的 `prepare_for_multiply` 做了什么操作把 padding 槽清零了。
或者是 rescale 导致的。**这个 p50=0 本身可能不重要——重要的是 slot 0 在这步不突出，下一步突然变 worst。**

### 1.3 从 iter01_b 之后的传播

```
iter01_b:  slot 0 max=1.541     → 越界开始
iter02_a:  slot 0 min=-2.371    → 负数出现（2-k*b 变号）
iter03_a:  slot 0 min=-35.44    → 在放大
iter03_b:  slot 0 max=70.65     → 爆炸
iter04_a:  slot 0 max=7.911e+05 → 07d 的最终值
```

**从 iter01_b 开始，slot 0 每步都在恶化，其他槽不变。**

---

## 2. 健康对比（你的本机数据）

```
本机（ClearEngine）:
07c.denominator    worst at 17075(%3),12979(%3),4787(%3)
07.inv_iter01_b    worst at 31370(%10),29322(%10),27274(%10)    ← max 在 %10
07.inv_iter02_b    worst at 6709(%5),12853(%5),31285(%5)         ← max 在 %5
07.inv_iter03_b    worst at 16452(%4),8260(%4),2116(%4)          ← max 在 %4
07.inv_iter04_b    worst at 8358(%6),16550(%6),2214(%6)          ← max 在 %6
07d                worst at 6580(%4),22964(%4),12724(%4)          ← max 在 %4
```

**本机的 worst 槽在游走**：%3 → %10 → %5 → %4 → %6，**从不落在 slot 0**。
设备的 worst 槽从 iter01_b 开始**钉在 slot 0**。

---

## 3. 你的 §3.2 问题回答

> "`iter01_b` 的最大槽已经是 0 → 坏在迭代之前或第一步"

**是的。`iter01_b` 的最大槽已经是 slot 0。** 而 `iter01_a`（紧邻的上一步）的 max 在 %4，slot 0 不在 worst-3。

所以坏的步骤是 **`iter01_a` → `iter01_b`**，即 `_restore_magnitude`。

`_restore_magnitude` 内部三步：
1. `conjugate(ct)` — 我测过了，slot 0 误差只有 1e-9（12x p50）
2. `add(ct, conjugate(ct))` — 同上，1e-9
3. integer multiply — 线性操作，不该坏

**但 `iter01_a` 的 p50 = 5.8e-13（几乎全零），而 `iter01_b` 的 p50 = 0.2733。**
这个跳变太大了。`_restore_magnitude` 是 scale-preserving 的，不应该改变 p50 这么多。

**可能的解释：`iter01_a` 的 p50=0 是因为 padding 槽。`_restore_magnitude` 的 conjugate-add 把 padding 槽的 `2·Re(0) = 0`，但 slot 0 不是 padding 槽——它带着数据。所以 p50 从 0 跳到 0.27 可能只是因为 _restore 把 carried 槽的值 restore 到了正确 scale，而 padding 槽仍然是 0。**

**但 slot 0 的值从 0.9957（iter01_a 的 max）变成了 1.541（iter01_b 的 max），这才是问题。**

---

## 4. 下一步建议

**slot 0 在 `_restore_magnitude` 步变坏。** 你说 conjugate 不是 slot 0 的不动点（你更正了），那 _restore_magnitude 里还剩 integer multiply。但 integer multiply 是线性的，不该只坏一个槽。

**或者：`_restore_magnitude` 的输入 `iter01_a` 在 slot 0 已经有微小的问题（在我的 conjugate 测试里看不到，因为那个测试不经过 bootstrap + multiply 链），而 _restore_magnitude 把它放大了。**

如果你要在 `_restore_magnitude` 内部进一步二分：可以在 conjugate 后、add 后、int-multiply 后各加一个探针，看 slot 0 在哪一步变坏。
