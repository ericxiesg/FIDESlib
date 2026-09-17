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
