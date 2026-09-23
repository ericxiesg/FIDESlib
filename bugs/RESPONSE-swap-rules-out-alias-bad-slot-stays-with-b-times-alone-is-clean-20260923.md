# 交换排除了别名：坏槽跟着 b 走，不跟调用顺序；_times 单独跑也是干净的

日期：2026-09-23。针对 `4f69d53`。**三个实验全跑了。**

---

## 0. 结论

| 问题 | 答案 |
|---|---|
| 别名（第一次 _times 破坏了 correction） | ❌ **排除**。换序后坏槽仍在 `b`，不跟到 `a` |
| `_times` 单独跑在 slot 0 有问题吗 | ❌ **没有**。test passed，slot 0 / p50 = 0.7x 和 1.2x |
| `_restore_magnitude` 是空操作吗 | ✅ **是**。`_times` 探针和 `_restore` 后探针逐位相同 |
| 坏槽是确定性的吗 | ✅ **是**。两次跑（swap 和 no-swap）`b_times` worst at 都是 0, 9891, 20170 |
| 那问题在哪 | **`_times(b, correction)` 在迭代里对 slot 0 算错，但单独跑没错**。可能是 bootstrap 后的密文有特定结构 |

---

## 1. SWAP 实验（决定性）

### 1.1 对比

| | no-swap `a_times` | no-swap `b_times` | swap `a_times` | swap `b_times` |
|---|---|---|---|---|
| max | 1.002 | **1.505** | 0.9977 | **1.484** |
| >1 | 1/32768 | **1/32768** | 0/32768 | **1/32768** |
| worst at | 31280(%0), 12848(%0), 4164(%4) | **0(%0)**, 9891(%3), 20170(%10) | 16848(%0), 436(%4), 12724(%4) | **0(%0)**, 9891(%3), 20170(%10) |

**换序后 `b_times` 的 worst at 完全不变**：0, 9891, 20170——三个槽一模一样。

而 `a_times` 在 swap 后变干净了（0/32768 >1），worst at 也变了（变成了 436/12724，正是分母最小的 %2048=436 族——健康的预期行为）。

### 1.2 解读

- **不是别名**：如果第一次 `_times` 破坏了 `correction`，换序后坏槽应该跟到 `a`。它没有。
- **是操作数决定的**：`b` 在 slot 0 的值和 `correction` 在 slot 0 的值相乘时算错。
- **确定性**：两次跑 slot 0/9891/20170 都是最差——不是随机噪声选中的，是数据结构决定的。

### 1.3 一个细节：no-swap 的 `a_times` 也有 1 个 >1

no-swap `a_times` max=1.002, `>1 1/32768`, worst at 31280(%0)。
swap `a_times` max=0.9977, `>1 0/32768`。

这说明 `a_times` 也有一个槽偶尔越界——但它在 %0（31280 = 15×2048 + 80），不是 slot 0。
swap 后这个越界消失了（因为 bootstrap 噪声不同）。**slot 0 的越界才是确定性的。**

---

## 2. `_times` 单独测试

```
b * correction  (level 36):
  slot 0      error 1.69235e-10   value 0.24073 against 0.24073
  slot 0 / p50 of the rest: 0.7x     ← slot 0 低于中位数！
  worst 30430(%16=14), 21389(%16=13), 15547(%16=11)

ones * correction  (level 36):
  slot 0      error 4.55231e-10   value 0.341074 against 0.341074
  slot 0 / p50 of the rest: 1.2x     ← 正常
  worst 15588(%16=4), 29665(%16=1), 19993(%16=9)
```

**`_times` 单独跑在 slot 0 完全正常。** 两个 operand pair 的 slot 0 误差都在 1e-10 级别，和 p50 一致。worst 槽散布在 %16 的各个位置。

**所以 `_times` 本身没有 slot 0 bug。** 问题只在实际迭代的上下文里出现。

---

## 3. `_restore_magnitude` 确认是空操作

```
iter01_a_times:  max 1.002   worst at 31280(%0), 12848(%0), 4164(%4)
iter01_a:        max 1.002   worst at 31280(%0), 12848(%0), 4164(%4)    ← 逐位相同
iter01_b_times:  max 1.505   worst at 0(%0), 9891(%3), 20170(%10)
iter01_b:        max 1.505   worst at 0(%0), 9891(%3), 20170(%10)       ← 逐位相同
```

**`_times` 后和 `_restore_magnitude` 后完全一样。** 确认你在 `38dbf00` 的分析：iter01 时 `b.delta=0.274`，`int(1/0.274/256)=0`，`_restore_magnitude` 什么都不做。

---

## 4. 综合分析

### 4.1 已排除

| 假设 | 排除方式 |
|---|---|
| `_restore_magnitude` 是坏步骤 | delta=0.274 → 空操作；探针逐位相同 |
| 别名（第一次 _times 破坏 correction） | 换序后坏槽仍在 b |
| `_times` 本身在 slot 0 有 bug | 单独测试 slot 0 / p50 = 0.7x~1.2x |
| bootstrap 误差有位置结构 | worst 行散布（之前的报告） |
| 共轭不动点 | 你已更正；源码验证 slot 0 → 65535 |
| `multiply/relinearize/rescale` 有 slot-0 特例 | 源码审查全是否定 |

### 4.2 剩下的

`_times(b, correction)` 在迭代里算错 slot 0，但：
- `_times` 单独跑没错
- `b` 和 `correction` 单独看都是好的
- 乘积超出映射值域（1.505 > 上确界 1.0）

**这指向 bootstrap 后的密文结构。** `b` 经过 bootstrap，它的密文内部表示（系数、模数链）可能和单独 encrypt 的密文不同。`_times` 对 bootstrap 后的密文做 multiply 时，可能在 slot 0 产生错误——即使同一个 `_times` 对 encrypt 的密文是正确的。

### 4.3 一个可以测的假设

test 里用的是 `engine.encrypt(b0)` 和 `engine.encrypt(correction)`——两个都是 fresh encrypt。
迭代里的 `b` 是 bootstrap 后的密文，`correction` 是 `subtract(scalar, bootstrap密文)` 的结果。

**可以改 test：让 `b` 经过 bootstrap 再做 `_times`，看 slot 0 是否变坏。**

---

## 5. 数据汇总

### 5.1 no-swap（baseline）

```
第一次 he_inv:
07.inv_iter01_a_times   max 1.002    >1 1/32768   worst at 31280(%0),12848(%0),4164(%4)
07.inv_iter01_b_times   max 1.505    >1 1/32768   worst at 0(%0),9891(%3),20170(%10)
07.inv_iter01_a         max 1.002    >1 1/32768   worst at 31280(%0),12848(%0),4164(%4)
07.inv_iter01_b         max 1.505    >1 1/32768   worst at 0(%0),9891(%3),20170(%10)

第二次 he_inv:
07.inv_iter01_a_times   max 1606                  worst at 18979(%3),23075(%3),10787(%3)
07.inv_iter01_b_times   max 4.929e+06             worst at 736(%0),17120(%0),21216(%0)
07.inv_iter01_a         max 1606                  worst at 18979(%3),23075(%3),10787(%3)
07.inv_iter01_b         max 4.929e+06             worst at 736(%0),17120(%0),21216(%0)
```

### 5.2 swap（THORFHE_SWAP_TIMES=1）

```
第一次 he_inv:
07.inv_iter01_a_times   max 0.9977   >1 0/32768   worst at 16848(%0),436(%4),12724(%4)
07.inv_iter01_b_times   max 1.484    >1 1/32768   worst at 0(%0),9891(%3),20170(%10)
07.inv_iter01_a         max 0.9977   >1 0/32768   worst at 16848(%0),436(%4),12724(%4)
07.inv_iter01_b         max 1.484    >1 1/32768   worst at 0(%0),9891(%3),20170(%10)

第二次 he_inv:
07.inv_iter01_a_times   max 2704                  worst at 26919(%7),24871(%7),16679(%7)
07.inv_iter01_b_times   max 1.274e+06             worst at 32681(%9),6057(%9),30633(%9)
07.inv_iter01_a         max 2704                  worst at 26919(%7),24871(%7),16679(%7)
07.inv_iter01_b         max 1.274e+06             worst at 32681(%9),6057(%9),30633(%9)
```

### 5.3 test_times_at_slot_zero

```
b * correction:      slot 0 / p50 = 0.7x   worst at 30430(%14)
ones * correction:   slot 0 / p50 = 1.2x   worst at 15588(%4)
```
