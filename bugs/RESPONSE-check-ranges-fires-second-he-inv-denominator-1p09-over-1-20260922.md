# --check-ranges 生效：he_exp 在窗口内，第二次 he_inv denominator max=1.09 > 1.0

日期：2026-09-22。针对 `a754d4b`。**热 cache + `--check-ranges`，EXIT=1（guard 拒绝）。**

---

## 0. `--check-ranges` 在 GPU 上生效了

协作者加的 `engine.inspectable = True`（通过 `--check-ranges` flag）让 range guards
在 GPU engine 上也工作了。6 次 he_exp range check + 1 次 he_inv range check 全部打印。

---

## 1. he_exp range checks：全部通过

```
[range] he_exp observed [-4.536, 10.203]  against window [-27.249, 21.727]  ✅
[range] he_exp observed [-10.101, 6.760]  against window [-27.249, 21.727]  ✅
[range] he_exp observed [-10.755, 3.848]  against window [-27.249, 21.727]  ✅
[range] he_exp observed [-5.454, 6.569]   against window [-27.249, 21.727]  ✅
[range] he_exp observed [-5.029, 6.573]   against window [-27.249, 21.727]  ✅
[range] he_exp observed [-4.736, 9.225]   against window [-27.249, 21.727]  ✅
```

**carried max=10.203**，在窗口 `[-27.25, 21.73]` 内。he_exp 没有问题。

注意：07a 的 carried max=38.1 是 **score**（he_exp 的输入是 `score / 32`，
38.1 / 32 = 1.19，但 range check 乘回 32 后是 `1.19 * 32 = 38.1`…不对，
observed max=10.203 对应 score=10.203 * 32 = 326？不，he_exp 的 `scale=32`，
`values = decrypt(x) * 32`，所以 observed 就是 score 单位）。

等等——07a carried max=38.1 但 he_exp observed max=10.203？**不是同一个量**。
07a 是 refresh 后的 scores（8 ct），he_exp 的 6 行对应 6 个 ciphertext 的
carried slots。38.1 是全部 8 ct 的 max，10.203 是其中 6 个 ct 的 carried max。

---

## 2. 第一次 he_inv：通过

```
[range] he_inv observed [0.081449, 1.0897] ratio 0.0747 against epsilon 0.046875
```

denominator 在 `[0.081, 1.09]`，ratio=0.075 > epsilon=0.047。**通过。**

但 max=1.0897 **已经超过 1.0**——Goldschmidt 的 `2 - k*b` 在 b>1 时变号。
这个 1.09 的槽是第一次 he_inv 里 `>1` 的那一个（之前探针 `>1 1/32768`）。

---

## 3. 第二次 he_inv：被 guard 拒绝

```
ValueError: he_inv: the denominator runs over [0.08145, 1.09] (median 0.1571)
on the 8448 slots that carry data, but the iteration is set up for [0.04688, 1].
```

**第二次 he_inv 的 denominator 和第一次几乎一样**（[0.081, 1.09]）。
guard 在 `inv_input_lift` 探针之前抛出，所以没有 `07d`、`07e`、`inv_input_lift`
的数据——程序直接退出了。

**关键**：第二次 he_inv 的 denominator max=1.09 > 1.0。这不是上次看到的 1e+12
（那是没有 `--check-ranges` 时，guard 没拦住，迭代跑完了的输出）。
**现在 guard 在迭代之前就拦住了**，所以我们看到了真正的输入：max=1.09。

---

## 4. 那 1e+12 是哪来的？

上次没 `--check-ranges` 时，07e 的 carried max=1.089e+12。
现在有 guard 拦住后，看到第二次 he_inv 的输入 max=1.09。

**1e+12 是 Goldschmidt 迭代放大的结果**，不是输入。输入 max=1.09，
迭代里 `a = a * (2 - b*a)` 在 b=1.09 时 `2 - 1.09 = 0.91`，a 变成 `1 * 0.91 = 0.91`……
等等，这不应该发散到 1e12。

**除非**：`--inverse-lift 3` 在第二次 he_inv 里把 denominator 乘了 3。
1.09 * 3 = 3.27 > 1，然后 `2 - 3.27 = -1.27`，a 变成 `1 * (-1.27) = -1.27`，
然后平方到 1.6，再乘到 5.2，再平方到 27……确实会发散。

**但 guard 看到的 denominator 是 1.09**（lift 后的值？还是 lift 前的？）。
如果 guard 在 lift 之前检查，那 lift 后是 1.09 * 3 = 3.27；
如果 guard 在 lift 之后检查，那 1.09 就是 lift 后的值。

---

## 5. 探针数据

```
07a0.score_refresh_input  carried max 0.319   padding max 0.3807    ← bootstrap 输入
07a.refreshed_scores      carried max 38.1    padding max 17        ← bootstrap 输出（119x 放大）
07b.exp                   max 0.144                                 ← exp 后正常
07c.denominator           max 0.3632                                ← 第一次分母
```

和上次一致。bootstrap 把 carried max 从 0.319 放大到 38.1。

---

## 6. 需要协作者看

1. **第二次 he_inv 的 denominator max=1.09 > 1.0**。第一次也是 1.09（同一个槽）。
   这个 >1 的槽是 `worst at slot 0`——它是 padding 还是活跃槽？guard 报告
   "8448 slots that carry data"，所以 **1.09 在活跃槽**。

2. **`--inverse-lift 3` 是否作用于第二次 he_inv？** 如果 lift 在 guard 之后，
   那 1.09 * 3 = 3.27 就是 Goldschmidt 看到的 b，必然发散。如果 lift 在 guard
   之前，1.09 就是 b，也会发散（`2 - 1.09 = 0.91`，a 从 1 变成 0.91，不发散——
   但 b 也在迭代，`b = b * (2 - b*a)²`，b=1.09, a=0.91 → `2-1.09*0.91 = 1.008`，
   b 变成 `1.09 * 1.008² = 1.108`，越来越远——确实发散）。

3. **denominator 的 max=1.09 是怎么来的？** 07c 的 max=0.3632，但第二次 he_inv
   的 denominator 是 `07d * exp(scores/2)` 求和。07d max=8.6e+05（第一次逆元
   的个别坏槽），乘以 exp 再求和可能产生 >1 的值。

4. **`inv_input_lift` 探针没出来**（guard 在探针之前抛了）。能否把探针移到
   guard 之前，或者 guard 改成 warning 而不是 exception？

---

## 7. timing

程序在第二次 he_inv 处退出（~50s 处），没有跑完。没有 accuracy/timing 数据。

完整日志：`/home/zhiyuan/bench-run/gpu_a754d4b.log`（128 行）
