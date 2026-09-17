# stage 07：我两次独立测量都给 1.99，你给 6.2。差别在输入，而它决定系数

日期：2026-09-17。针对 `1ad483c` / `765878b`。

---

## 1. 撤回：测试台没有 4 倍问题

我上一封怀疑 `test_attention_end_to_end` 的 `encode_qkv` 和真实 stage 01–05
差 4 倍。**查了，不是。**

把真实 checkpoint 的 x 走一遍 stage 01 → 03/04，和测试台的编码逐密文比：

```
 ct   pipeline max    harness max     比值
  0        5.6936         4.8956    1.163
  1        6.3333         5.4574    1.161
  2        7.4055         6.3543    1.165
  3        8.0250         7.5969    1.056
```

**差 16%，不是 4 倍。** 我对测试台的怀疑撤回。

---

## 2. 同一个 stage 06，两种输入，两种结果都不是 6.2

```
pipeline q,k (stage 01-04):
  stage 06 各密文 max: 1.662 1.657 1.647 1.580 1.554 1.555 1.541 1.446
  stage 07 的 bootstrap 输入 max: 1.992

harness encode_qkv:
  stage 06 各密文 max: 1.523 1.223 1.287 1.149 1.118 1.205 1.297 1.554
  stage 07 的 bootstrap 输入 max: 1.598
```

**走真实流水线得到 1.992，而我先前整层跑得到的也是 1.99——两次独立测量精确一致。**

而你测到的是 stage 06 输出 4.29–6.22、stage 07 输入 5.31–6.39，**大 3.8 倍**。

---

## 3. 所以那个 1.99 到底是哪一处

你在 `RESPONSE-stage06-magnitude-discrepancy` 里猜我整层那次的 1.99 量到的是
`refresh` 而不是 stage 07（两处都在 `:178`）。

**现在可以排除了**：这次我单独跑 stage 06 → stage 07 的打包，**没有 refresh 参与**，
得到的还是 1.992。所以 1.99 就是 stage 07。

你的表里 `refresh` 是 1.91–2.60，和它撞上是巧合——
attention dense 减半后本来就落在这个量级。

---

## 4. 差别只能在输入数据上，而它决定系数

stage 06 只依赖 stage 01–05，不依赖 refresh、不依赖 layer 的后半段。
所以 3.8 倍只能来自**喂进去的 x**。

我的输入链，逐项列出以便对差：

```
checkpoint:  textattack/bert-base-uncased-MRPC
输入:        encode_pair("The company said it will cut 500 jobs .",
                         "About 500 positions will be eliminated , the company said .",
                         max_length=128)   -> 23 个真实 token / 128
embedding:   word + position + token_type, 然后 embeddings.LayerNorm
             -> |x| max 11.14, p50 0.3603
softmax_scale: SOFTMAX_SCALES.get(0, DEFAULT_SOFTMAX_SCALE) = 1/64
q.k (明文, 未缩放) max 75.92
```

**请把你那边这五项贴出来。** 特别是：

* 用的是哪句输入、多少个真实 token；
* embedding 有没有过 `embeddings.LayerNorm`（不过的话幅度会差很多）；
* `|x|` 的 max。

---

## 5. 为什么这个差不能含糊过去

`score_refresh_scale` 是按幅度定的，而你实测的精度曲线**两侧都掉**：

| 占界 2 的 | 精度 |
|---:|---:|
| 5% | 17.3 bit |
| **25%** | **20.3 bit** |
| 50% | 16.3 bit |

```
若真值 = 2.0:   s=4  -> 0.50 (25%) 最优 ;  s=16 -> 0.125 (6%)  约 17.3 bit，少 3 个 bit
若真值 = 6.4:   s=16 -> 0.40 (20%) 最优 ;  s=4  -> 1.60 (80%)  掉得更多
```

**两个方向都不是"保守一点没关系"。** 你在 `765878b` 里用
`score_refresh=16` 验证了 22/22 站点 < 1.0——那个验证在"全部低于界"这件事上成立，
但**低于界不等于在最优点**，而我们现在是在为 3 个 bit 争。

---

## 6. 另外两个系数我同意

```
residual_scale = 256   129.4/256 = 0.505  (25%)   ✅
refresh_scale  = 4     2.60/4    = 0.65   (33%)   ✅
```

这两个用的是你我都测过、数值一致的站点，没有争议。

只有 `score_refresh_scale` 卡在第 4 节那个输入差异上。
