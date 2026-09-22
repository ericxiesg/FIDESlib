# 新探针定位：bootstrap 放大 score 100x，07e 第二次分母 carried max=1e+12

日期：2026-09-21。针对 `aec4c9e`。**热 cache 61.2s，EXIT=0。**

---

## 0. 新探针给出的链路

```
07a0.score_refresh_input    carried max  0.319       padding max 0.3807     ← bootstrap 输入，正常
07a.refreshed_scores        carried max  38          padding max 16.04      ← bootstrap 输出，放大 100x
07b.exp                     max 0.2015                                     ← exp 后正常
07c.denominator             max 0.3169                                     ← 第一次分母，正常
07d.inverse_denominator     max 8.631e+05            >1 1/32768             ← 第一次逆元，基本正常
07e.halved_denominator_k384 carried max  1.089e+12   padding max 3.648e-05 ← 第二次分母，活跃槽爆了
```

---

## 1. Bootstrap 放大了 score

| | 07a0（输入） | 07a（输出） | 倍数 |
|---|---|---|---|
| carried max | 0.319 | 38 | **119x** |
| padding max | 0.3807 | 16.04 | 42x |
| p50 | 0.02713 | 1.014 | 37x |

**bootstrap 把 score 从 0.38 放大到 38。** 这远超 he_exp 的窗口 `[-27.25, 21.73]`——
但 `_check_exp_range` 没触发（fideslib engine 的 `inspectable` 可能是 False，或
`check_ranges` 没开）。协作者加的 guard 逻辑是对的，但在 GPU engine 上没生效。

**`worst at` 位置**：07a 的最差槽在 `196608(%0), 163840(%0), 229376(%0)`——
slot 0 的倍数（`%0`），即每个 group 的第一个 slot。**carried max 38 在活跃槽，
不是 padding。**

---

## 2. 07e：第二次 he_inv 的输入就已经爆了

```
07e.halved_denominator_k384  level 7  carried max 1.089e+12  padding max 3.648e-05
  worst at 28672(%0), 6144(%0), 30720(%0)
```

**k=384**，`worst at` 全在 `%0`（group 边界槽）。

第二次 he_inv 的 `b`（denominator）在进迭代前就是 1e+12，而 epsilon 是
`precision / 128 / 2`——分母远超 1，Goldschmidt 必然发散。

**`07e` 是 `07d` 乘以 exp(scores/2) 再求和得到的**——07d 的 max=8.6e+05
已经不对了，平方后乘以 exp 再求和就到了 1e+12。

---

## 3. 第一次 he_inv（正常收敛）

PADDING_FLOOR + `--inverse-lift 3` 下，第一次 he_inv 完全正常：

```
iter01_b  min +0.0519  max +1.608   <0 0/32768     >1 1/32768
iter02_b  min -1.213   max +0.0476  <0 2/32768     >1 0/32768
iter03_b  min -0.032   max +56.67   <0 1/32768     >1 1/32768
iter04_b  min -1.015e6 max +0.0039  <0 2/32768     >1 0/32768
→ 07d = iter04_a: max 8.631e+05  (>1 1/32768, worst at slot 0)
```

iter03_b 的 max=56.67 是个别槽（p99=0.003724），不影响整体。
07d 的 max=8.6e+05 也只有 1 个槽 >1（worst at slot 0）。

---

## 4. 第二次 he_inv（从 iter01 发散）

```
iter01_b  min -4.239e+06  max +7.765e+06   <0 12416  >1 20336  worst at 14932(%4),21076(%4),31316(%4)
iter02_b  min -5.93e+13   max +4.745e+13   <0 17504  >1 15264
iter03_b  min -1.114e+29  max +7.863e+28   <0 16001  >1 16767
```

和上次完全一致：1e6 → 1e13 → 1e28 → ... → 1e214。

---

## 5. LayerNorm（11b）

```
11a.variance     level 17  max 8.841e+07   >1 2048/32768 (padding)
11b.inverse_sqrt level 10  max 2.317e+154  <0 16278  >1 16490  worst at 21494(%6),12750(%14),30726(%6)
```

和上次一致。

---

## 6. `_check_exp_range` 没触发

日志里没有 `[range]` 行，也没有 `ValueError`。协作者加的 guard 在 GPU engine
上没生效——可能是 `inspectable` 属性为 False，或 `check_ranges` 没设。
**这个 guard 在 ClearEngine 上应该能工作，但 fideslib engine 需要确认。**

---

## 7. 需要协作者看

1. **bootstrap 为什么把 score 放大 100x？** 07a0→07a：0.38→38。这是 score refresh
   bootstrap 的正常行为还是 bug？如果 `score-refresh-scale=16` 的作用是乘 16，
   那 0.38 × 16 = 6.08，不到 38。38 / 0.38 ≈ 100，接近 `residual_scale=256 / score_refresh_scale=16 * ...`?

2. **07e 的 k=384 是什么？** `halved_denominator_k384` 的 k 是 `update_inv_D` 里的整数
   scalar。k=384 被平方后乘进分母，384² = 147456，这本身不应该导致 1e+12。

3. **`_check_exp_range` 在 GPU engine 上需要什么才能生效？** 需要设 `check_ranges=True`
   和 `engine.inspectable=True`？

---

## 8. timing

```
total  61.22s  (hot cache, 691 fields loaded, 0 encoded)
```

完整日志：`/home/zhiyuan/bench-run/gpu_aec4c9e_full.log`（222 行）
