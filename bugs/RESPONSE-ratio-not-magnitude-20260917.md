# 输入不是差别（q.k 两边都是 99.46）。差别是密文装的是 S 还是 4S

日期：2026-09-17。针对 `0b0dc87`。

---

## 1. 你关于 padding 的那一条是对的，我的 75.92 是错的口径

我先前算 `q.k max` 时只取了真实 token（`[:n]`）。算上 padding：

```
只算真实 token    q.k max 75.92   at head=10 pos=21,22
全部 128 个位置   q.k max 99.46   at head=3  pos=116,115     <- 和你逐项一致
```

head=3、(116,115)、99.46——**和你的完全相同**。你说"padding 的 embedding
与句子无关，所以 q.k max 与输入无关"，这一条成立，我确认了。

**所以输入不是我们的分歧所在。** 我那个"请贴五项"的请求可以撤回。

---

## 2. 但分歧还在，而且现在能指到一个槽

`4 * q.k * scale = 4 * 99.46 / 64 = 6.2162` 是**明文推算**。
问题是密文里装的是不是这个数。

我把那个明文最大值定位到它在打包里应该在的位置，然后解密那一个槽：

```
明文 S（已含 softmax_scale）max = 1.554   at head=3 (tau=116, j=115)
diagonal = (115 - 116) mod 128 = 127  ->  ct 7, group 15
密文 ct7 在 slot(15, 116, 3) 的值 = 1.4056
比值 密文/明文 = 0.9045
```

**密文装的是 `S`（约 0.90 倍），不是 `4S`。**

而且这和 ct7 的整体最大值一致（1.446）——不是"最大值在别处、这个槽恰好小"。

---

## 3. 这意味着什么

`4 * q.k * scale` 和你实测的 6.22 吻合，是因为**两边都是 4S**；
我实测的是 **0.9S**。同一份代码、同一个 q.k、差 4.4 倍。

所以要查的不再是"谁的输入不同"，而是**你那边的 stage 06 为什么给出 4S**。

**`layer.py:36` 的注释说的是 "`he_softmax` sees `4 * (q.k) * scale`"——
注意是 `he_softmax`，不是 stage 06。** `he_softmax` 看到的是 stage 07 的输出，
而 stage 07 的拆包会把 stage 06 的结果**加倍**。所以按注释推，
stage 06 应当装 `2S`，`he_softmax` 看到 `4S`。

我实测 stage 06 装 0.9S（注释预测 2S，差 2.2 倍）；
你实测 stage 06 装 4S（注释预测 2S，差 2 倍，方向相反）。

**两边都和注释对不上，而且偏向相反。** 这条注释大概不能当依据。

---

## 4. 我把判据做进命令里了

`python -m thorfhe.bench magnitudes --through 06` 现在除了幅度，还报**比值**：

```
checkpoint  textattack/bert-base-uncased-MRPC
input       23/128 tokens, LayerNorm'd embedding |x| max 11.14  p50 0.2755
scales      softmax 0.015625  residual 1.0  refresh 1.0  score_refresh 1.0

stage 06 outputs      1.662  1.657  1.647  1.58  1.554  1.555  1.541  1.446
stage 07 would refresh 1.869  1.992  1.786  1.662

plaintext score (with softmax_scale) max 1.554 at head 3, (116, 115) - diagonal 127
the ciphertext holds 1.4056 there, a ratio of 0.9045
```

**幅度不能跨机器比（依赖输入），比值可以。**

请在你那边跑同一条命令，贴最后两行。

* 你也得到 **0.90** → 那么你先前那个 6.22 的测量方式有问题，而正确的系数是
  `score_refresh_scale = 4`（1.99 / 4 = 0.50，占界 2 的 25%，最优点）；
* 你得到 **4.0** → 那么是我们两台机器上的代码或数据真的不同，那要单独查，
  因为同一个 commit 不该给出两个答案。

`--through 06` 只跑到注意力分数，内存需求很小——这台 8 GB 的机器跑得动，
所以两边一定能跑同一条。
