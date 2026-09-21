# 最大的那个分数在 **padding** 里——探针现在会告诉你是哪一种

日期：2026-09-21。再更正我自己一处，并给出下一步要看的那一行。

---

## 0. 又一处更正

我在 `9add352` 的护栏注释里写：

> 每个槽都检查，不只是承载槽：`he_softmax` 在指数**之后**才 mask，
> 所以 padding 也会被多项式求值，**padding 的越界和真实槽的越界一样会发散**。

**后半句是错的。** 看 `softmax.py:150-151`：

```python
exp_u = [self.he_exp(ct, ...) for ct in normalised]
exp_u = [self.rescale(self.multiply(ct, mask)) for ct, mask in zip(exp_u, attention_mask)]
```

padding 槽的指数值**下一行就被乘以 0**。它离窗口再远也进不了分母。
按我原来那样查全槽，会因为一个**马上要被乘零**的槽而拒绝整次运行。

已改：护栏拿到下一行那个 mask，**只判决活下来的槽**。

---

## 1. 本机实测：最大的分数确实在 padding 里

探针现在会拆开报：

```
07a0.score_refresh_input   min -0.3199  max +0.3807   carried max 0.319   padding max 0.3807
07a.refreshed_scores       min -10.41   max +12.43    carried max 10.21   padding max 12.43
```

**12.43 是 padding 槽的，真实 token 只到 10.21。**

而 he_exp 实际被判决的量程（只算活下来的槽）：

```
[range] he_exp observed [-4.37353, 9.58181] against window [-27.2493, 21.7269]
[range] he_exp observed [-9.56423, 6.85416] ...
[range] he_exp observed [-10.2008, 3.75546] ...
```

**真正要紧的槽只到 9.58**，离窗口上界 21.73 还有 **2.3 倍**。
（我之前按全槽算成 1.78 倍，也是偏严的。）

`relRMSE` 不变，仍是 8.479e-04。

---

## 2. 所以你那个 37.5 要先分类

`--per-stage` 现在会在 `07a` 那行末尾多打两个数：

```
07a.refreshed_scores  ... carried max <A>  padding max <B>
```

| 情况 | 含义 | 下一步 |
|---|---|---|
| **B ≈ 37.5，A ≲ 10** | 越界的是 padding，**下一行就乘零了，无害** | 窗口这条线**划掉**，回去查 `07a0` 和 `07d` |
| **A ≈ 37.5** | 真实 token 的分数越界，最大那项偏低 46% | 窗口是真问题，要重标定或查为什么分数变大 |

**在拿到这两个数之前，不要为窗口改任何东西。** 我已经在这条线上错了两次
（"会起飞"→ 其实是衰减；"padding 也要查"→ 其实无害），所以这次先量后说。

---

## 3. 不受影响、仍然建议的

| | 状态 |
|---|---|
| `07a0.score_refresh_input` | 仍然是关键：`scores` 准到 3e-7 而 `07a` 大 3 倍，中间只有那一次 bootstrap |
| `--score-refresh-scale 2` | 仍然建议。那是 bootstrap 信噪比（输入只用了量程的 2.4%），和窗口无关 |
| 第一次 he_inv 已修好 | 不变 |

---

## 4. 请跑的还是那一条

```bash
THORFHE_DEBUG=1 python3 -m thorfhe.bench fhe --engine fideslib --device cuda:0 \
  --depth 37 --dnum 4 --bootstrap-level-budget 3,3 --binary-rotations \
  --refresh-after-dense --residual-scale 256 --refresh-scale 4 \
  --score-refresh-scale 2 --layers 1 --limit 1 --compact --per-stage \
  --extra-rotation-keys 6 --rotation-max-steps 4 --inverse-lift 3 \
  --plaintext-cache <cache>
```

要四个数：`07a0` 的 min/max、`07a` 的 **carried max** 和 **padding max**。

221 passed, 67 skipped。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
