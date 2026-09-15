# EasyFHE 的 THOR 参考实现：参数与调度对照

日期：2026-09-15。源：`../easyfhe-examples/thor/`（Part16 引作"已验证可跑通"的那份）。
凡标 **[实测]** 的是在本机跑出来的。

---

## 一、参数对照

`easyfhe-examples/thor/config.py` 把整组参数写成常量，抄录如下：

| | 我们 | EasyFHE |
|---|---|---|
| log N / slots | 16 / 2^15 | 16 / 2^15 ✓ |
| **Δ（rescale 素数位宽）** | **2^50** | **2^59** |
| **q0（首素数）** | **2^55** | **2^60** |
| **depth** | **37** | **30** |
| **dnum** | **4** | **3** |
| bootstrap level budget | (3,3) | (3,3) ✓ |
| **bootstrap 落点** | **depth − 17 = 20** | **14** |
| secret key | SPARSE_TERNARY | SPARSE_TERNARY ✓ |
| **rescale policy** | **FIXEDMANUAL** | **`"manual"`** |
| 输入 limb | — | 10 |
| 验收阈值 | — | rel-L2 ≤ 5e-2 |

### 1.1 参考实现用的就是 FIXEDMANUAL

`RESCALE_POLICY = "manual"`、`SCALE_MODE = "fixed"`。

这一条值得单独说，因为 Part16 的 5.1 建议"临时切 FIXEDAUTO 跑一遍"来缩小范围。
**那是往偏离已知可用点的方向走。** FIXEDMANUAL 不是我们的异想天开，是这条电路被验证过的配置；
换成 FIXEDAUTO 得到的信息是"另一个配置下会怎样"，不是"这个配置为什么不行"。

### 1.2 对得上的部分

我们的常数和它逐个吻合，说明两边移植的是同一份 THOR：

* `softmax_epsilon` = 2^-11，层 2 用 2^-18 → 我们的 `NARROW`/`WIDE` 的 `inv_epsilon` 一致；
* `ff_layernorm_min_var/max_var` = 0.2/150，层 9 和 10 用 0.75/2500 → 和我们的
  `he_layernorm2` / `he_layernorm3` **完全一致**（也说明我之前"窗口宽了 16 倍、可以收紧"
  那条要谨慎：参考实现保留了 THOR 的宽窗口）。

### 1.3 对不上的一处：attention key scale

它用 THOR 原值 **1/512**（层 2 用 1/1024），我们用 **1/64**（层 2 用 1/128）。

我们文档里写了理由（`layer.py` 的 `SOFTMAX_SCALES` 注释：这是我们自己的 stage 量出来的值），
但**参考实现站在 THOR 那一边**。如果之后要对齐参数，这一项要一起看，
因为它直接决定 `he_exp` 的输入落在拟合窗口的哪里。

---

## 二、调度：它在 A·V 之后刷新，我们不

`easyfhe-examples/thor/model/bert.py`，`run_encoder_layer` 的主干：

```python
attention_context = calculate_attention_context(...)                       # A·V
attention_context = [bootstrap_cipher(c, ctx, bp) for c in attention_context]   # ← 刷新
dense_output      = attention_dense(...)                                   # stage 10
attention_residual = add(residual, dense_output)
attention_norm    = attention_layernorm(..., bootstrap_program, ...)       # 内部自举
dense1            = ff_dense1(...)
dense1            = bootstrap_dense1_pairs(...)                            # GELU 前
activated         = gelu(dense1, ctx)
dense2            = ff_dense2(...)
_, residual       = ff_residual_add_and_bootstrap(...)                     # stage 15
return feed_forward_layernorm(..., bootstrap_program, ...)                 # 内部自举
```

这和 THOR 笔记的 ⑧ 一致，和我们不同——我们把刷新推到了 stage 10 之后（`refresh_after_dense`）。

### 2.1 我据此提的改法，实测是错的 [实测]

我在 `thor-note-conformance-20260914.md` §3.2 提过"把刷新挪到 ⑧ 之后可省 2 次 bootstrap"。
加了 `refresh_after_context` 开关实测：

```
refresh 在 dense 之后（现状）  depth 37 → OK（22 次 bootstrap）；36 / 33 / 30 → 失败
refresh 在 context 之后         depth 37 / 36 / 33 / 30 → 全部失败
```

**搬过来更差，不是更好。**

### 2.2 原因：那是两个耦合的选择，不是一个

参考实现**同时**做了两件事：

1. 在 A·V 之后刷新；
2. **两个 LayerNorm 都内部自举**（`attention_layernorm` 和 `feed_forward_layernorm`
   都收 `bootstrap_program` 参数）。

我们的 LayerNorm **从不自举**——因为我们刷新后有 20 格，够它跑完（见
`thor-note-conformance` §3.3，我当时还把这条列为"我们更省"）。
**只搬第 1 条不搬第 2 条，从刷新点到 LayerNorm 的那条链就比任何试过的 depth 都长。**

这也顺带解释了一件我们一直没解释的事：**它 bootstrap 只落到 14 就够用，我们最长段要 19。**
不是它的电路更省，是它在更多地方刷新。

---

## 三、这对下一步意味着什么

**参数对齐（Δ 2^50→2^59、depth 37→30、dnum 4→3）不是一个独立旋钮。**
depth 30 配 bootstrap 落 14，在我们**当前的刷新布局下跑不通**——上面 §2.1 那张表里
depth 30 两种布局都失败。要用它那组参数，得连它的刷新布局一起搬，
也就是**让 LayerNorm 内部自举**。

所以顺序应该是：

1. 先编译验证 c2 的两个改动（Part16 的第 1、2 条）——那是结构性 bug，和参数无关；
2. 如果还有精度问题，**再**考虑参数，而且要**成套搬**：
   Δ=2^59 + q0=2^60 + dnum=3 + depth=30 + bootstrap→14 + LayerNorm 内部自举 + key scale 1/512；
3. 单独调其中任何一项，都可能像 §2.1 一样比现状更差。

**我不建议现在动参数。** 理由和 Part16 自己第 3 节的理由一样：
在结构性 bug 修掉之前换参数，会把两个问题混在一起。
