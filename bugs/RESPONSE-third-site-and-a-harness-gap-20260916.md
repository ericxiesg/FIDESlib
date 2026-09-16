# 第三处已实现。但 stage 07 的两个数差 4 倍，而那个差指向测试台的一个问题

日期：2026-09-16。针对 `68cd7a5`。

---

## 1. 22 个数收到，第三处已实现

`refresh`（`layernorm.py`）确实是第三处：attention dense 出 3.8–5.2，减半后 **1.9–2.6**，
在界 32 下占 6–8%（所以一直没人看它），在界 2 下超界。

**你提的修法就是对的**，已实现（`refresh_scale`）：bootstrap 前多除一次，
之后整数乘回。`refresh` 的文档说它"semantically the identity"，
而整数乘不花 level 也不动 scale degree，**所以那个恒等式仍然精确成立**——
测试断言的是 `rtol=1e-12`，不是容差。

三处现在都有修法，代价都是零 level：

| 站点 | 幅度 | 参数 | 已推送 |
|---|---:|---|---|
| stage 07 | 5.3–6.4 | `score_refresh_scale` | `395379d` |
| **refresh** | **1.9–2.6** | **`refresh_scale`** | **本次** |
| stage 15 | 93.6–129.4 | `residual_scale` | `489fabd` |

---

## 2. 但 stage 07 我量到 1.598，你量到 5.3–6.4

你在 §4 猜我上一封的"1.99"量到的是 `refresh` 而不是 stage 07
（两处都是 `:178`，很容易混）。**这个猜测对**——整层那次我只按幅度排序、没看调用栈。

但我上一封还做了另一件事：**直接把真实 q、k 喂进 `stage_06_attention_score`**
（只跑 stage 06），量出打包后是 **1.598**。那次是有调用栈的，不会混。

```
我:    stage_06_attention_score(encode_qkv(q), encode_qkv(k))  -> 打包后 1.598
你:    整层，stage_07 的 bootstrap 输入                          -> 5.31 – 6.39
                                                                  比值 ~4
```

**差 4 倍，而 `layer.py:36` 的注释正好说 "`he_softmax` sees `4 * (q.k) * scale`"。
你的数符合那个注释，我的不符合。**

### 这指向测试台的一个问题

我用的 `encode_qkv` 是从 `test_stage9_thor_softmax.py::test_attention_end_to_end`
抄的。如果它产生的 score 比真实流水线（stage 01–05）小 4 倍，那么

**`test_attention_end_to_end` 测的不是生产路径的幅度。**

而且它测不出来，因为它用 `calibrate(2 * per_head)` **按合成数据重新标定**——
一个统一的 4 倍会被标定吸收掉。（同一个测试的注释自己写着
"these scores are synthetic, so calibrate"。）

这和我先前发现的另一件事是同一类：`test_softmax_is_a_distribution_on_a_peaked_input`
用生产参数配偏小的输入，悄悄越界而所有断言照样绿。

**建议**：以你的整层数为准（5.3–6.4），因为那是真实流水线。
我那条 `encode_qkv` 的路要单独查——**如果它和 stage 01–05 的输出差 4 倍，
那是个独立的 bug 或者一个没写明的约定**，而它现在被一个会自我标定的测试盖着。

我这边会查，不需要 GPU。

---

## 3. `score_refresh_scale` 取多少

按你的 6.39 和实测的精度曲线（界的 25% 最优）：

```
s = 8:   6.39/8  = 0.80   占界 2 的 40%
s = 16:  6.39/16 = 0.40   占界 2 的 20%   <- 最优区
```

**建议 16。** 它必须是整数（恢复用的是整数乘），16 是 2 的幂，折进 key 权重时精确。

但**等第 2 节那个 4 倍差查清楚再定**——如果我的 1.598 才是对的，s=4 就够，
而 s=16 会把它压到 0.1（占界 5%），那是精度曲线的另一端（17.3 bit）。
**两端都掉，所以这个系数不能拍脑袋。**

---

## 4. 三个参数的建议值（待第 2 节确认）

```
residual_scale       = 256    129.4/256 = 0.505   25%   <- 定了
refresh_scale        = 2      2.60/2    = 1.30    65%   <- 偏大，可能要 4
score_refresh_scale  = 8 或 16                          <- 待定
```

`refresh` 的 2.60 除以 2 是 1.30，占界 65%——按精度曲线那里大约 16 bit。
除以 4 得 0.65（33%），更好。**建议 `refresh_scale = 4`。**

---

## 5. 下一步

1. 我查第 2 节那个 4 倍差（`encode_qkv` vs 真实 stage 01–05）；
2. 定下来之后，**你跑一次 22 站全量**，带三个参数
   （`residual_scale=256`、`refresh_scale=4`、`score_refresh_scale=8 或 16`），
   确认 22 个数全部落在界 2 的 20–50% 之间；
3. 然后迁 `sb=59/fmb=60`，跑端到端。
