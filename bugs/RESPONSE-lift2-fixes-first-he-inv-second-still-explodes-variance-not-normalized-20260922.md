# lift 2 修了第一次 he_inv，第二次仍爆；variance 8.4e7 没归一化；bootstrap 误差 0.065

日期：2026-09-22。针对 `698eca4`。**三个跑完了（A/B/C）。**

---

## A: `--inverse-lift 2 --check-ranges warn`（58.8s，EXIT=0）

### range checks

```
he_exp  6 checks  全部通过（carried max 10.37 < window 21.73）         ✅
he_inv  #1        observed [0.051, 0.621]  epsilon 0.031              ✅ 通过！
he_inv  #2        observed [-1.9e+14, 8.0e+30]                         ❌ 爆了
he_invsqrt #1     observed [6.55e+07, 8.41e+07]  epsilon 0.081        ❌ 爆了
he_invsqrt #2     observed [6.10e+09, 7.77e+09]  epsilon 0.033        ❌ 爆了
```

### 探针

```
07.inv_input_lift2   (第一次)  carried max 0.6531    padding max 0.5649    ✅
07d.inverse_denominator        max 2.497e+15  >1 2/32768  worst at 0(%0)   ❌ 2个坏槽
07e.halved_denominator_k256    carried max 4.019e+30                          ❌ 爆了
07.inv_input_lift2   (第二次)  carried max 1916                               ❌ 输入就坏了
11b.inverse_sqrt               max 2.228e+154                                 ❌
```

### accuracy: encrypted 0%

**第一次 he_inv 修好了**（max 0.621 < 1.0）。但 07d 有 2 个槽 >1（max=2.5e+15，
worst at slot 0），这 2 个坏槽通过 `update_inv_D` 传播到第二次 he_inv 的输入，
爆到 1e+30。

---

## B: `--inverse-lift 3 --check-ranges warn`（~60s，EXIT=0）

### range checks

```
he_inv  #1        observed [0.084, 0.960]  epsilon 0.047               ✅ 通过（勉强，0.96 逼近 1）
he_inv  #2        observed [0.062, 1.23e+12]                            ❌ 爆了
he_invsqrt #1     observed [6.70e+07, 8.79e+07]                         ❌
he_invsqrt #2     observed [6.10e+09, 7.77e+09]                         ❌
```

### 探针

```
07.inv_input_lift3   (第一次)  carried max 0.9817    padding max 0.5628   ✅ 但逼近 1
07d.inverse_denominator        max 7.919e+05  >1 1/32768  worst at 0(%0)  ❌ 1个坏槽
07e.halved_denominator_k384    carried max 4.099e+11                        ❌ 爆了
07.inv_input_lift3   (第二次)  carried max 2243                             ❌ 输入就坏了
```

### A vs B 对照

| | A (lift 2) | B (lift 3) |
|---|---|---|
| 第一次 he_inv max | 0.621 ✅ | 0.960 ✅（勉强） |
| 07d max | 2.5e+15 (2 slots >1) | 7.9e+05 (1 slot >1) |
| 07e carried max | 4.0e+30 | 4.1e+11 |
| 第二次 he_inv 输入 | 1916 | 2243 |

**lift 3 的第一次 he_inv 更接近 1（0.96 vs 0.62），但 07d 的坏槽数量更少（1 vs 2）。**
两次的第二次 he_inv 都爆了——07d 的坏槽通过 `update_inv_D` 传播。

---

## C: pytest `test_how_wrong_the_bootstrap_is_on_a_full_vector`（48.6s，2 passed）

```
bootstrap absolute error over 32768 slots:
  max   0.0649213
  p99   0.0401588
  p50   0.0105397
  -> 1/D relative error would be about 2.34
```

**bootstrap 在满向量上的绝对误差 max=0.065，p99=0.040。**
协作者之前用的 0.017 是 single-slot 估计，满向量 max 是 0.065（4x 更大）。

---

## 核心问题

### 1. 07d 的坏槽（slot 0）从哪来？

第一次 he_inv 的输入 max=0.62（lift 2）或 0.98（lift 3），都在 1 以内。
但 07d 的输出有 1-2 个槽 >1，max 到 2.5e+15 或 7.9e+05。

**worst at slot 0**——这不是 padding 槽（padding 的 worst 在其他位置）。
slot 0 是第一个活跃槽。bootstrap 误差在 slot 0 上可能特别大？

### 2. LayerNorm variance 没归一化

```
he_invsqrt observed [6.55e+07, 8.41e+07]  against [0.081, 1]
```

variance 在 8.4e+07，但 iteration 要求 `[0.081, 1]`。**variance 没有被
归一化到 [0, 1] 范围**。协作者推了 per-layer variance windows (`42a43ed`)，
但 variance 值本身在 8.4e+07——不是 window 的问题，是 **scale 的问题**。

### 3. `update_inv_D` 传播坏槽

07d 的 1-2 个坏槽（slot 0）通过 `update_inv_D` 传播到第二次 he_inv 的输入。
07e = `07d² × k × exp(scores/2)` 的求和——一个 2.5e+15 的槽平方后到 6.3e+30，
乘以 k=256 后到 1.6e+33，再求和就到 4e+30。

**第二次 he_inv 的输入爆炸不是第二次 he_inv 的问题，是 07d 的坏槽传播的。**

---

## 需要协作者看

1. **07d slot 0 为什么坏？** 第一次 he_inv 输入 max=0.62（正常），但输出 slot 0
   到 2.5e+15。这是 Goldschmidt 在某个中间迭代 step 上 slot 0 越界的？
   bootstrap 误差 max=0.065，如果集中在 slot 0，那 0.62 + 0.065 = 0.685，
   不到 1。但迭代中 `a*(2-b*a)` 如果 b 在某步 >1，a 就会变负。

2. **variance 8.4e+07 需要什么 scale？** `VARIANT_BOUNDS` 是 `[0.081, 1]`，
   但实际 variance 在 8.4e+07。需要除以 ~1e8 才能进窗口。`per-layer variance windows`
   是改窗口还是改 scale？

3. **`update_inv_D` 能不能 clip 07d 的坏槽？** 如果 07d 只有 1-2 个槽 >1，
   能不能在传给 `update_inv_D` 之前把 >1 的槽 clamp 到 1？
