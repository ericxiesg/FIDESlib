# stage 06 = 4 * q.k * (1/64) = 6.22，公式精确匹配；差异在 q.k 本身

日期：2026-09-17。针对 `6a018ad`。

---

## 1. 五项数据

```
checkpoint:  textattack/bert-base-uncased-MRPC
输入:        bench 默认 row 0 = "He said the foodservice pie business doesn 't fit
             the company 's long-term growth strategy ." /
             "The foodservice pie business does not fit our long-term growth strategy ."
             -> 44 个真实 token / 128
embedding:   word + position + token_type, 然后 embeddings.LayerNorm ✅
|x| max:     11.14
|x| p50:     0.30
softmax_scale: SOFTMAX_SCALES.get(0, DEFAULT_SOFTMAX_SCALE) = 1/64 ✅ (和你的相同)
q.k (明文, 未缩放, per-head) max:  99.46
```

**你用的 "500 jobs" 句子不是 bench 默认输入。** bench 用的是 `nyu-mll/glue` mrpc
validation 的 row 0。但——见第 3 节——**两个句子的 q.k max 完全相同**。

---

## 2. FHE stage 06 = 4 * q.k_unscaled * softmax_scale，精确匹配

```
q.k_unscaled max = 99.46   (q_h @ k_h^T，未除 sqrt(d))
4 * 99.46 * (1/64) = 6.2162
实测 stage 06 输出 max = 6.22   ✅ 精确匹配
```

**4x 来自输入加倍（q 和 k 各 2x）。softmax_scale 折进了 key 权重。**

---

## 3. q.k max 来自 padding，与输入句子无关

```
max score at: head=3, position=(116, 115)
real tokens:  44  -> 位置 44-127 是 padding
Is padding?   i=True, j=True   ← 两个都是 padding
```

**q.k max 来自两个 padding 位置的注意力。** 因为所有 padding token 的
embedding 相同（word_embeddings[0] + position_embeddings[t] + token_type_embeddings[0]），
所以 **不同输入句子的 q.k max 完全一样**。

验证：bench 输入（44 token）和你的 "500 jobs"（23 token）：

```
bench:    q.k_unscaled max = 99.46
500 jobs: q.k_unscaled max = 99.46   ← 完全相同
```

**所以输入句子的差异不是 3.8 倍差的来源。**

---

## 4. 我的 q.k = 99.46 vs 你的 75.92，差 1.31 倍

```
我的 q.k_unscaled max = 99.46    (per-head: q_h @ k_h^T，12 heads × 128 × 128)
你的 q.k_unscaled max = 75.92
比值 = 1.31
```

可能的原因：

1. **padding 处理不同**：我的 max 包含 padding-padding 位置。去掉 padding，
   real-only max = 10.20（scaled）= 81.6（unscaled），和你的 75.92 只差 7%。
   **你的 75.92 可能是 real-only，我的 99.46 包含 padding。**

2. **per-head vs global**：如果你算的是 `q @ k^T`（不拆 head），结果会不同。

---

## 5. 关键：THOR 不 mask score，padding 分数走 bootstrap

bench.py:421 的注释：

> "THOR masks the padding inside the softmax rather than with a -10000 in the score,
>  so the real token count has to reach stage 07"

**所以 padding 位置的 score（包括 max=99.46 对应的那对）确实走 stage 07 的 bootstrap。**
这意味着 **stage 07 的 bootstrap 输入 max = 6.22 是真实的生产值**，
不是测试台假象——它来自 padding-padding 注意力，而且对所有输入都一样。

---

## 6. stage 07 bootstrap 输入 = 6.22，不是 1.99

```
stage 06 输出 max = 6.22   (8 个密文，全是实数)
stage 07 bootstrap 输入 max = 6.39   (复数打包后)
```

你测到的 1.99 可能是：
- **没走完整 stage 01-05**（即使 q/k 幅度只差 16%，stage 06 的 streamed packing
  可能在你的测试里没有还原生产路径的缩放）
- **或者你的 stage 06 测试用了不同的 geometry/packing**

但公式 `4 * q.k * (1/64)` 精确给出 6.22，与整层实测一致。

---

## 7. 结论：score_refresh_scale = 16

```
stage 07 bootstrap 输入 = 6.39
score_refresh_scale = 16  ->  6.39/16 = 0.40  (占界 2 的 20%)  ✅ 最优
```

如果用 s=4：6.39/4 = 1.60（80%），精度掉到 ~12 bit。
如果用 s=8：6.39/8 = 0.80（40%），精度 ~16 bit。

**s=16 是唯一落在 25% 最优区的选择。**

---

## 8. 请确认

你那边的 stage 06 单独测试是否：
1. 用了和 bench 相同的 geometry（THOR_BERT）？
2. q/k 是从 stage 01-05 走出来的（不是 encode_qkv）？
3. stage 06 输出是否包含 padding 位置？

如果三点都是，应该得到 6.22 而不是 1.66。如果有一点不是，那就是差异来源。
