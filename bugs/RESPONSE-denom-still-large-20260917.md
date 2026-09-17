# query masking 后 denom 仍超界：real query 行也有 241

日期：2026-09-17。针对 `881f6fe`。

---

## 1. 现象

`bench fhe --engine clear` 在 `he_inv` 报 `denom [0, 344.4]`，超出
`inv_epsilon=2^-14` 的下界 `6.1e-5`。

---

## 2. query masking 生效了，但不够

检查 padding_mask 结构（23 real tokens）：

```
mask[i]: nonzero=2976, query_real AND key_real
query_real=5888, key_real=5888, both_real=3968
mask==1 matches both_real: True (mask 是对的)
```

**mask 正确地把 padding query AND padding key 都置零了。**

---

## 3. 但 real query 行的 denom 也大

在 he_inv 被调用的那一刻解密 denominator：

```
denom: nonzero=8448, |denom|>1: 1360
large values at tau positions: min=1, max=43
at padding query (tau>=23): 672, max=344.37
at real query (tau<23):      688, max=241.89   ← real query 也有大 denom
```

**688 个 real query slot 的 denom > 1，最大 241.9。**

---

## 4. 为什么 real query 的 denom 大

denominator = sum over key positions of exp(score * scale)。
attention_mask 把 padding key 位置置零了，所以 real query 行的 denom 应该是
`sum_{j=0}^{22} exp(score[i,j] * scale)`，最多 23 个项。

但 **23 个 exp 项的和可以很大**——如果 score 大、scale 大，exp 可能到几十。
新的 `softmax_scale = 1/16`（修正后）比旧的 `1/64` 大 4 倍，exp 的值也跟着大。

这是 `inv_epsilon` 和 `softmax_scale` 之间的耦合：scale 改大 4 倍后，
denom 的分布范围也变了。

---

## 5. inv_epsilon 的下界

当前 `inv_epsilon = 2^-14 = 6.1e-5`，对应 denom 上界 `1/epsilon = 16384`。
但 `_check_inversion_range` 的逻辑是检查 denom 在 `[epsilon, 1]` 区间内。

**denom=241 远超 1**，所以报错。但 Goldschmidt 可以处理 denom > 1
（只要 `1/denom` 不太小）。

问题可能在 `_check_inversion_range` 的上界——它假设 denom < 1，
但修正后的 softmax scale 让 denom 可以到几百。

---

## 6. 选项

1. **放宽 `_check_inversion_range` 的上界**到 1000 或更高
2. **在 he_softmax 里对 denom 做缩放**（除以一个常数，inv 后乘回）
3. **调 `inv_epsilon`** 到能容纳 denom=241 的范围

请定。
