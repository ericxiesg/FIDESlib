# 那个 241 不是 `he_inv` 的界太紧，是送进指数的 score 大了 4 倍

日期：2026-09-17。针对 `12f412b`。

---

## 0. 先回答你的三个选项

| 选项 | 结论 |
|---|---|
| 1. 放宽 `_check_inversion_range` 的上界到 1000 | **不行。** 上界 1 是收敛条件，不是风格检查 |
| 2. 在 `he_softmax` 里缩放 denom | 不需要——真正的原因修掉之后 denom 自己就落回界内 |
| 3. 调 `inv_epsilon` | **下界要调**（见 §4），但不是为了容纳 241 |

你在 §5 写的"Goldschmidt 可以处理 denom > 1（只要 1/denom 不太小）"——**不行**。
第一步的 `k = 2/(epsilon+1) ≈ 2`，`correction = 2/k - D = 1 - D`，
D>1 时 correction 变号，之后每一步平方发散。实测（纯 numpy，精确算术）：

```
D=  0.0001  ->  a=     9987     (= 1/D, 对)
D=     0.5  ->  a=    1.997     对
D=       1  ->  a=   0.9987     临界
D=       2  ->  a=     -inf
D=     241  ->  a=     -inf
```

`softmax.calibrate` 本来就写着 `if largest > 1.0: raise`，理由是同一个。

---

## 1. 根因：`DEFAULT_SOFTMAX_SCALE = 1/16` 大了 4 倍

`he_softmax` 看到的必须是 BERT 自己的 attention score，因为 softmax 不是尺度不变的。
数三个因子：

1. 投影给出 `s * (x @ W.T) + 2b`——`encode_weight` 把权重减半，bias 在减半和
   `y + conj(y)` 之间加进去，所以 bias 不随 `s` 走。`s = 2` 时这是 **2 倍**的 BERT `q`；
2. stage 06 精确携带乘积；
3. `stage_07_softmax` 的 bootstrap fold 再乘 2。

`s` 通过 q 和 k **各进来一次**，所以 `he_softmax` 看到 `2 * s^2 * (q.k) * scale = 8 * (q.k) * scale`。
匹配 BERT 的 `(q.k)/sqrt(64)` 要 **`scale = 1/64`**——也就是改坏之前的值。

### 是什么掩盖了这个因子

`bench magnitudes --through 06` 用 `encode_activations(g, x)` 进层，**振幅是 1，不是 2**
（`bench fhe` 走的是 `args.output_scale * hidden`，默认 2.0，`bench.py:431`）。
`s = 1` 时同一个投影给出 `x @ W.T + 2b`，对着这个参考比，比值正好读出 1.0000——
于是 stage 06 看起来携带 `(q.k) * scale`，而真实流水线携带的是它的 4 倍。

我上一轮就是照这个 1.0000 把 1/64 改成了 1/16。是我改错的。

---

## 2. 复现：534 是从明文算出来的，四位有效数字

MRPC validation row 0（44 tokens），layer 0，纯 numpy，不需要 GPU 也不需要密文：

```
score 乘数    he_softmax 看到的 max      denom max      denom min
  x1  (对)              12.43          5.861e-04      4.197e-05
  x2                    24.86          3.082e-02
  x4  (现在)            49.73          5.3436e+02     3.679e-05    <- 你报的 [3.679e-05, 534.4]
  x8                    99.46          7.368e+09
```

然后在 `ClearEngine` 上跑 stage 01→06，按 `out[ct][g,tau,b]` 的约定把整个
12×128×128 的 score 张量解出来，逐元素比：

```
进层振幅 2.0（bench fhe 的默认）:
  stage 06 / (q.k * scale)，bias x1  : 中位 4.00000  [3.99998, 4.00001]
  he_softmax 输入 / BERT score       : 中位 4.00000  [3.99998, 4.00001]
  由它算出的 denom                    : [3.679e-05, 534.4]
```

**和你 `he_inv` 报的区间一模一样。**

改回 `scale = 1/64` 之后同一条链：`he_softmax 输入 / BERT score = 1.000000`。

---

## 3. 已改

* `layer.py`：新增 `ACTIVATION_SCALE = 2.0`（THOR 的"每个密文携带两倍"不变量，
  现在是一个具名常量，因为有两处从它推导）；`DEFAULT_SOFTMAX_SCALE = 1/64`，
  `SOFTMAX_SCALES = {2: 1/128}`，注释写了完整的因子账。
* `bench.py`：`--output-scale` 的默认值取自 `ACTIVATION_SCALE`；
  **`command_magnitudes` 现在也按 `--output-scale` 进层**，并且它的明文参考改成
  `s * (x @ W.T) + 2b`。这两处不一致就是这轮的起因，所以两边现在测的是同一条流水线。
* `test_qkv_computes_xw_plus_bias` 按进层振幅参数化（1 和 2）——单点钉住时
  `x @ W.T + 2b` 是对的但会误导，两点钉住才能看出 `s` 只乘在第一项上。
* 新增 `test_stage_06_hands_the_softmax_berts_own_score`：跑 stage 01→06（真实
  `encode_layer`，随机权重，THOR 几何），断言 `2 * stage06 == BERT score`。
  这是唯一能抓住这个因子的测试形态——per-stage fidelity 会按最佳拟合缩放掉它，
  `calibrate` 会把它吸进窗口，每个 stage 测试都用自己合成的输入。

---

## 4. 下界：`inv_epsilon` 现在的值覆盖不住真实数据

`scale = 1/64` 之后 denom 全部落回界内，**上界到处都成立**（最大 0.184，layer 8）。
但下界不成立。明文扫 64 个 validation 样本、12 层（denom 就是 BERT score 过
多项式的和，不需要密文）：

```
  layer   score min  score max   window                 min D      max D   需要   现在
  0          -11.06      11.52   [-27.2, 21.7]      3.365e-05  8.997e-04   2^-15  2^-14
  1          -15.08      15.86   [-27.2, 21.7]      2.500e-05  1.148e-02   2^-16  2^-14
  2          -33.25      32.44   [-70.0, 70.0]      1.813e-05  3.441e-03   2^-16  2^-18
  3          -12.33      18.34   [-27.2, 21.7]      4.147e-05  1.792e-02   2^-15  2^-14
  4          -10.87      15.71   [-27.2, 21.7]      3.156e-05  5.758e-03   2^-15  2^-14
  5          -15.78      14.98   [-27.2, 21.7]      3.299e-05  4.800e-03   2^-15  2^-14
  6          -13.89      14.67   [-27.2, 21.7]      2.543e-05  4.783e-03   2^-16  2^-14
  7          -14.17      16.41   [-27.2, 21.7]      4.186e-05  7.016e-03   2^-15  2^-14
  8          -13.78      21.48   [-27.2, 21.7]      2.264e-05  1.844e-01   2^-16  2^-14
  9          -17.44      18.96   [-27.2, 21.7]      1.163e-05  2.522e-02   2^-17  2^-14
  10         -16.97      11.54   [-27.2, 21.7]      9.871e-06  8.989e-04   2^-17  2^-14
  11         -16.09      11.54   [-27.2, 21.7]      1.115e-05  1.772e-03   2^-17  2^-14
```

iteration 数：`2^-14 -> 9`，`2^-16 -> 10`，`2^-17 -> 11`。**照实测取值要多 2 个 level，
我们没有。** 这一条我还在做（见 §6），先不改代码。

---

## 5. 顺带两件要记的事

**layer 8 的 score max 是 21.48，窗口上沿是 21.73**——64 个样本已经占到 99%。
`calibrate` 的 docstring 说"超出范围 10% 就足以让分母溢出"。全量 408 个样本大概率
会越过去。这台机器只缓存了 64 行（`hub.rows` 的 cache 是 `complete: false`），
**你那边能跑全量的话请跑一次，只要 12 层的 score min/max 和 denom min/max**，
不需要密文。

**padding query 行。** `881f6fe` 之后 `padding_mask` 也掩掉了 query 侧，所以那些行的
denom 是 0——而 `_check_inversion_range` 检查的是 `ones` 标的所有 slot，0 也在下界外。
你 §3 量到"at padding query: 672 个，max=344.37"说明你那次跑的时候 query 掩码没生效
（或者是 `881f6fe` 之前的 build）。这两件事要一起定：`ones` 也应该按 token 数收窄，
padding 行才不会带着垃圾穿过后面的 bootstrap。

---

## 6. 下界的修法：把每层的窗口中心挪下去，不花 level

denom 的绝对高度由窗口中心 `mid` 定——`he_exp` 只用 `mid` 做平移，`D` 随 `exp(-mid/2)` 走——
而 `min/max` 的比值和 `mid` 无关。所以把 `mid` 下移到"max D 贴到界"，`epsilon` 就变成 `min/max`。

代价是多项式在更大的 `|t|` 上求值，而它只是 exp 的 minimax 拟合，所以**精度是量出来的**：
明文跑完整的 `he_softmax` 代数（多项式、`n` 次平方、归一、`l` 次平方再归一），
对真实 softmax 比最大绝对误差。32 个样本，12 层，取 max D <= 0.5：

```
  layer   shift      min D    max D   eps  iters      error     现在 iters/error
  0      -15.26    0.02128    0.466    -6      5   1.49e-06        10 / 5.39e-06
  1      -10.26   0.001219   0.4881   -10      7   2.87e-06        10 / 5.68e-06
  2      -19.75    0.00353   0.4798    -9      7   5.34e-04        10 / 3.18e-03
  3       -9.26   0.001364   0.4623   -10      7   7.37e-07        10 / 2.24e-06
  4      -11.76   0.003134   0.4867    -9      7   8.63e-07        10 / 3.11e-06
  5      -12.01   0.003942   0.4896    -8      6   3.37e-06        10 / 3.23e-06
  6      -12.01   0.002594   0.4514    -9      7   1.23e-06        10 / 2.70e-06
  7      -11.26   0.003377   0.4606    -9      7   1.68e-06        10 / 1.97e-06
  8       -4.51  5.431e-05   0.4424   -15     10   1.63e-06        10 / 2.87e-06
  9       -8.51  0.0002342    0.447   -13      9   1.25e-06        11 / 4.05e-06
  10     -15.26   0.005576   0.4656    -8      6   3.51e-06        11 / 4.82e-06
  11     -14.01    0.00309   0.4913    -9      7   7.08e-07        11 / 3.68e-06

  he_inv 在 12 层里一共: 85 levels 重新居中后, 123 按实测下界不动窗口
```

三件事值得注意：

1. **精度是变好的，不是变差**。每一层的 softmax 误差都降了，layer 2（wide 多项式，
   一直是最差的那个）从 3.2e-3 降到 5.3e-4，好 6 倍。
2. **max D 落在 0.44–0.49**，也就是 bootstrap 消息界 2 的 22–25%——正好是你三点测出来的
   精度峰值（25% 处 20.3 bit，50% 处掉到 16.3）。这不是我凑的，是 `target=0.5` 的直接结果。
3. **layer 8 挪不动**。它的 `min/max` 比值是 1.2e-4，跟中心无关，所以无论怎么居中都要
   10 次迭代——比现在的 9 次多一个 level。但比"不动窗口、按实测取下界"的 11 次少两个。

所以这条路把"多 2 个 level"变成"多 1 个 level，而且 12 层里只有 layer 8 需要"。

### 还没做

这要给 `he_exp` 加一个显式的 `shift` 参数（现在它写死取 `(min_x+max_x)/2`），
并把 12 层的 `(shift, inv_epsilon)` 做成一张表。我先不动——**这轮先把 scale 落地**，
因为那是纯粹的错，而窗口表是个设计选择，值得你先看一眼数字再定。

两件想请你确认的：

* **全量 408 行的 score min/max 和 denom min/max**（明文，12 层，不需要密文）。
  我这里只缓存了 64 行，而 layer 8 的 score max 已经到 21.48 / 窗口上沿 21.73。
* **`ones` 也要按 token 数收窄**。query 侧掩掉之后 padding 行的 denom 是 0，
  而 Goldschmidt 对 0 会把 `1/D` 推向无穷——那些行不被下游读，但要穿过 bootstrap。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

---

## 7. 补：窗口表已落地（承接 §6）

`4068f20` 之后又做了：

* `he_exp` 的拟合中心从窗口中点里**分离出来**，成了显式的 `shift`（默认仍是中点，
  所以 `calibrate` 和现有调用都不变）。分开是因为两者干的事不同：`max_x` 选多项式，
  中心决定 denom 落在哪里，而 Goldschmidt 的价钱是按 denom 的下端算的。
* `calibrate(scores, target=0.5)`：往下走中心，直到最大的 denom 刚好还在 target 以下，
  返回对应的 `shift` 和 `inv_epsilon`。**这张表就是它产出来的**，不是我手填的。
* `Softmax.LAYERS`：12 行 `(shift, inv_epsilon)`，64 个 MRPC validation 样本上量的。
  `stage_07_softmax` 现在按层查这张表。

```
layer   shift      min D     max D   levels        不动中心
  0    -15.26    1.74e-2     0.466      5      10  (error 差 4 倍)
  1    -10.26    1.06e-3     0.488      7      10
  2    -19.75    2.54e-3     0.480      7      10  (error 差 6 倍)
  3     -9.26    1.07e-3     0.462      7      10
  4    -11.51    2.51e-3     0.457      7      10
  5    -12.01    3.37e-3     0.490      7      10
  6    -12.01    2.59e-3     0.488      7      10
  7    -11.26    2.93e-3     0.492      7      10
  8     -4.51    5.43e-5     0.442     10      10
  9     -8.51    2.06e-4     0.447      9      11
 10    -15.26    5.11e-3     0.466      6      11
 11    -14.01    3.09e-3     0.491      7      11
                                       86     123
```

**layer 8 要 10 次迭代**，比现在的 9 次多一个 level。这是没办法的：它的
min/max 比值是 1.2e-4，和中心无关。但注意 **NARROW 的 2^-14 在 layer 8 上本来就是错的**——
实测最小值 5.43e-5 floor 到 2^-15，在 2^-14 之下。所以不是"多花一个 level"，
是"原来那个 level 数本来就不够，只是没人发现，因为 `he_inv` 在界外是饱和不是报错"。

两个测试：
* `test_recentring_the_fit_buys_iterations_without_moving_the_softmax`——同一组 score
  跑两次，中心差 13.5，迭代 9 次 vs 4 次，误差 6.04e-4 vs 6.44e-4。**持平**，
  这就是要钉的东西（level 省下来是算术，softmax 还是不是 softmax 才是要测的）。
* `test_the_layer_table_is_the_one_calibrate_produces`——表的内部一致性和 86 这个总数。

### 还是要你那边跑全量

这张表是 408 行里的 64 行量的（本机 dataset cache 只有 64 行，`complete: false`）。
`target=0.5` 给上界留了 2 倍，`inv_epsilon` floor 到 2 的幂给下界留了 1.8 倍（layer 8）
到 3.5 倍。**请在全量上重跑一次 `calibrate(scores, target=0.5)`**，每层一次调用，
输入是 `traces[sample][f"layer_{i}"]["scores_unmasked"][:, :tokens, :tokens]`。
数不对就把表换掉。

