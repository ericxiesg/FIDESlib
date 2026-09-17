# 三个修复已推，请在你那边跑两件本机跑不了的事

日期：2026-09-17。承接 `4068f20` / `b2813d1` / `3b06b7d`。

---

## 1. 已推的三个 commit

| commit | 内容 |
|---|---|
| `4068f20` | `SOFTMAX_SCALES` 改回 1/64——`ACTIVATION_SCALE` 通过 q 和 k **各进一次**，是平方进 score 的 |
| `b2813d1` | 每层单独的拟合中心 + `inv_epsilon`，`he_inv` 从 123 个 level 降到 86 |
| `3b06b7d` | `ones` 按 token 数收窄，padding 行的 `1/0` 不再穿过 bootstrap |

三个都有测试，而且**都是去掉修复就会红**的测试。特别是第三个：把 `_carrying`
退回旧行为，新测试报的是

```
he_inv: the denominator runs over [0, 0.4798] (median 0) on the 24576 slots that carry data
```

**和你报的 `[0, 344.4]` 是同一个形状**——median 为 0 就是所有 padding 行。
你 §3 量到"at padding query: 672 个, max=344.37"也是这个：query 掩码把那些行的 denom
置成了 0，而 `he_inv` 被要求求 `1/0`。

---

## 2. 请跑：`bench fhe --engine clear`

这是 `he_inv` 那三个问题修完之后的第一次整层跑，而**本机跑不了**：
`encode_layer` 要把 3072x768 的 feed-forward 权重编码成 32768-slot 的明文，
8 GB 装不下（同样的原因，`test_stage13_thor_gelu_dense.py` 在本机也跑不起来，
它和这三个 commit 无关，本来就跑不了）。

```
python -m thorfhe.bench fhe --engine clear --layers 1 --limit 8
python -m thorfhe.bench fhe --engine clear --layers 12 --limit 8     # 如果单层过了
```

要看的是：`he_inv` 不再报范围、per-stage fidelity、以及 12 层那跑的准确率
——那是第一次真正意义上的 MRPC 数字。

**注意 layer 8 现在要 10 次 Goldschmidt 迭代**（以前是 9）。如果 12 层那跑在 layer 8
的 level 上挂掉，那不是回归：旧的 9 次对 layer 8 本来就不够（实测最小 denom 是
5.43e-5，floor 到 2^-15，在原来给它的 2^-14 之下），只是 `he_inv` 在界外是饱和不是报错，
所以一直没人看见。要多一个 level 就报上来，我们再想办法。

---

## 3. 请跑：全量 408 行重新标定

`Softmax.LAYERS` 那张表是 **64 行**量的（本机 dataset cache 只有 64 行，`complete: false`）。
表是 `calibrate` 产出来的，不是手填的，所以重做就是每层一次调用：

```python
from thorfhe.softmax import calibrate
for index in range(12):
    scores = traces[sample][f"layer_{index}"]["scores_unmasked"][:, :tokens, :tokens]
    print(index, calibrate(scores, target=0.5))       # 需要把所有 sample 的 scores 拼起来
```

`calibrate(scores, target=0.5)` 会把中心往下走到"最大的 denom 刚好还在 0.5 以下"，
并返回对应的 `shift` 和 `inv_epsilon`。把 12 行数报回来，不一致我就换表。

**一个已知的风险点**：layer 8 的 score max 在 64 个样本上已经到 **21.48**，
而 NARROW 的窗口上沿是 21.73。`calibrate` 的 docstring 说超出范围 10% 就足以让分母溢出。
全量上如果它越过去了，`calibrate` 会直接 raise（`denominator reaches ...`），
那说明 layer 8 要走 wide 多项式，而不是只挪中心。

---

## 4. 顺便：`bench magnitudes --through 06` 现在应该报 1.0000

`command_magnitudes` 以前用 `encode_activations(g, x)` 进层，振幅是 1，
而 `bench fhe` 用 `args.output_scale * hidden`，默认 2.0。**这就是这一整轮的起因**——
振幅 1 下投影给出 `x @ W.T + 2b`，对着那个参考比正好读出 1.0000，
于是 stage 06 看起来携带 `(q.k) * scale`，而真实流水线携带的是它的 4 倍。

现在两边都按 `--output-scale` 进层，参考也改成 `s * (x @ W.T) + 2b`。
麻烦确认一下比值还是 1.0000——如果不是，说明还有第三处不一致。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

---

## 5. 更正：level 的节省是 21，不是 37

`b2813d1` 的 commit message 里我写的"86 levels 对 123"是**每层第一次 `he_inv` 的迭代次数**。
那些确实是那一次调用内部花掉的 level，但**不是按 1:1 记到层的深度上的**——
每次 `he_inv` 开头都有一个 bootstrap。

在 `ClearEngine` 上按 stage 07 输出密文回来的 level 实测：

```
  layer    now  before  delta
  0         11      13     -2
  1         11      13     -2
  2         13      17     -4
  3         11      13     -2
  4         11      13     -2
  5         11      13     -2
  6         11      13     -2
  7         11      13     -2
  8         14      13     +1     <- 唯一变贵的
  9         13      13     +0
  10        11      13     -2
  11        11      13     -2
          139     160     -21
```

**所以是 21 个 level，不是 37。** 定 depth 的时候按这张表，别按迭代次数。
layer 8 那 +1 是实的——如果 12 层跑在它身上挂掉，就是这一个 level。


---

## 6. 好消息：`2eda264` 的 22 站测量不用重跑

`softmax_scale` 从 1/16 改回 1/64，stage 07 的 bootstrap 输入差 4 倍，
所以第一反应是那 22 个站的幅度全要重量。**不用。**

你在 `RESPONSE-stage06-magnitude-discrepancy-20260916.md` 里报的 stage 06 输出是
`4.286 - 6.216`，其中 `ct[7] = 6.216`。而

```
stage 06 max = s^2 * (q.k)_max * softmax_scale = 4 * 99.46 * (1/64) = 6.2162
                                      1/16 的话 = 4 * 99.46 * (1/16) = 24.865
```

**6.216 对上的是 1/64，不是 1/16**（小数点后三位都对）。所以那一轮量的就是现在恢复的这个 scale，
`(residual=256, refresh=4, score_refresh=16)` 这三个系数照旧有效：

```
stage 07 站点 = 6.39 / 16 = 0.40   -> 界 2 的 20%
```

这也和你报的"瓶颈是 `he_inv` 的 0.971 而不是 stage 07"一致——如果那一跑是 1/16，
stage 07 会是 2.2，早就顶在最前面了。

**一个要留意的交互**：每层重新居中之后，`he_softmax` 里第一个 `he_inv` 的分母
从 0.18 提到 0.44-0.49，它自己的 bootstrap 输入跟着提。0.49 是界 2 的 24.5%，
正好落在你测出来的精度峰值（25% 处 20.3 bit）上，所以这是往好处走的；
但它是个新的站点位置，**重跑 22 站的时候留意一下**。


---

## 7. 第四个：layer 2 的 `softmax_scale` 也是错的（已改）

`SOFTMAX_SCALES = {2: scale/2}`——"layer 2 keeps THOR's factor of two"。这条在我们的单位里是错的。

两条多项式路径落在**同一个指数**上：`he_exp1` 配 `l=2` 和 `he_exp2` 配 `l=4`
都给出 `exp(u)`（`thorfhe.softmax` 的模块 docstring 就是这么推的）。既然都要 `u`，
把其中一条的 score 减半就是**把那一层的温度加倍**。

真实 checkpoint，layer 2 的 wide 路径对 BERT 自己的 softmax，最大绝对误差：

```
喂 BERT 的 score        (scale 1/64)   中心 0.00     0.0029
喂 BERT 的 score        (scale 1/64)   中心 -19.75   0.000524
喂一半                  (scale 1/128)  中心 0.00     0.612     <- 温度错
喂一半                  (scale 1/128)  中心 -19.75   0.613     <- 温度错
喂两倍                  (scale 1/32)   中心 0.00     0.600     <- 温度错
```

**0.61**——和 4 倍 score 那个错（0.908）是同一个量级。已改成
`SOFTMAX_SCALES = {}`，所有层都用 1/64。

新测试 `test_the_wide_path_wants_the_same_score_the_narrow_one_does`：在合成的
±32 score 上跑 wide + l=4，对真 softmax 比，实测 6.8e-3（均匀分布比真实分布难，
真实 layer 2 是 2.9e-3 / 5.2e-4）。界设在 1e-2，是为了把它和"温度错"那两个数量级分开。

**对你那边的影响**：layer 2 的 stage 06 输出**翻倍**（scale 从 1/128 变 1/64），
所以 layer 2 的 stage 07 bootstrap 站点幅度也翻倍。22 站那张表是 layer 0 量的，
layer 2 没单独量过——重跑站点幅度的时候把 layer 2 也带上。

