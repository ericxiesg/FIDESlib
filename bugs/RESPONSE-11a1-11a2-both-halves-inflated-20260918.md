# 11a1/11a2 探针：不是灾难性相消，是两个加数都被膨胀了 1e8 倍

日期：2026-09-18。承接 `44c5312`。**更正 `f63b733` 的灾难性相消诊断。**

---

## 0. 结论

协作者说对了：**不是灾难性相消**。11a1 和 11a2 差 200 倍，减法只去掉 0.5%。

**真正的问题：两个加数本身在 GPU 上都比 ClearEngine 大 ~1e8 倍。**

| 探针 | GPU | ClearEngine | 倍数 |
|---|---|---|---|
| 11a1.n_sum_of_squares | max **4.76e+07** | max 0.2161 | **2.2e8** |
| 11a2.squared_total | max **2.34e+05** | max 0.001773 | **1.3e8** |
| 11a.variance | max **4.755e+07** | max 0.2161 | 2.2e8 |

variance ≈ 11a1，因为 11a1 >> 11a2（200:1），减法几乎不减。**问题在 11a1 本身**：
`n * Σ(x²)` 在 GPU 上是 4.76e7，应该是 0.2161。

---

## 1. 实验条件

```
python3 -m thorfhe.bench fhe --engine fideslib --device cuda:0 --depth 37 --dnum 4 \
  --bootstrap-level-budget 3,3 --binary-rotations --refresh-after-dense \
  --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --layers 1 --limit 1 --per-stage --compact \
  --extra-rotation-keys 6 --rotation-max-steps 4
```

代码：`44c5312`（包含 11a1/11a2 探针）。

---

## 2. 完整探针数据

### Stage 07 (softmax)
```
07a.refreshed_scores      8 ct  level 17  max +37.73     p99 6.597
07b.exp                  8 ct  level 9   max +0.1465     p99 0.002677
07c.denominator          1 ct  level 9   max +0.2925     p99 0.176
07d.inverse_denominator  1 ct  level 15  max +5.298e+16  p99 0.2381   ← outlier
```

### Stage 10-11 (attention dense → residual → LN1)
```
10.attention_dense       8 ct  level 1   max +1.285e+19  p99 6.841e+18
11.residual              8 ct  level 20  max +9.348e+04  p99 5.362e+04
11a1.n_sum_of_squares    1 ct  level 17  max +4.76e+07   p99 4.349e+07  ← 应 0.2161
11a2.squared_total       1 ct  level 17  max +2.34e+05   p99 8.555e+04  ← 应 0.001773
11a.variance             1 ct  level 17  max +4.755e+07  p99 4.348e+07
11b.inverse_sqrt         1 ct  level 8   max +1.87e+124  p99 1.113e+124
```

### 第二轮 (layer 1 input)
```
11a1.n_sum_of_squares    1 ct  level 14  max +3.12e+09   p99 2.956e+09
11a2.squared_total       1 ct  level 14                  p99
11a.variance             1 ct  level 14  max +3.12e+09   p99 2.956e+09
11b.inverse_sqrt         1 ct  level 3   max +1.029e+49  p99 6.197e+48
```

---

## 3. 分析

### 3.1 不是相消

11a1 / 11a2 = 4.76e7 / 2.34e5 = **203x**（GPU），vs 0.2161 / 0.001773 = **122x**（ClearEngine）。
比例相近，减法不构成相消。协作者的 condition number 1.008 是对的。

### 3.2 是 sum_of_squares 累积膨胀

`n * Σ(x²)` 在 GPU 上是 4.76e7，ClearEngine 是 0.2161。**差 2.2e8 倍**。

`(Σx)²` 在 GPU 上是 2.34e5，ClearEngine 是 0.001773。**差 1.3e8 倍**。

两个都膨胀了，但 11a1 膨胀更多（2.2e8 vs 1.3e8）。

### 3.3 可能原因

输入 `11.residual` 的值在 GPU 上是 ~1e4 量级。ClearEngine 上应该小很多（values mask
已经预除以 `sqrt((1.05*max_var + var_e) * n^2)`）。

如果 residual 的**有效值**在 GPU 上和 ClearEngine 一样小，但 **FHE 噪声**让某些 slot
的值大了 1e4 倍，那么：
- `x²` 的噪声项会被平方放大
- `Σ(x²)` 会累积这些放大后的噪声
- `n * Σ(x²)` 再乘以 n

但 11.residual 的 p50 是 1.408e+04，p99 是 5.362e+04——这些不像是噪声，像是真实值。

**另一个可能**：values mask 的预除没有在 GPU 上正确执行。如果 mask 的编码或乘法在 FHE
下有误差，residual 就不会被正确缩放。

### 3.4 07d.inverse_denominator 的 outlier

21 密钥下 `07d.inverse_denominator` 的 max 是 5.3e+16（p99 只有 0.2381）。少数 slot
有巨大误差。这在 15 密钥下不存在。可能是 meet-in-the-middle 分解在某些 slot 上引入
了额外噪声。

---

## 4. 更正之前的诊断

`f63b733` 报告说"variance = n\*Σx² − (Σx)² 在 GPU 上灾难性相消"。**这是错的**。
两个加数差 200 倍，不构成相消。真正的问题是两个加数本身都被膨胀了 ~1e8 倍。

`1413375` 报告说"variance 灾难性相消依然存在"。**也是错的**，同样原因。

---

## 5. 下一步

1. **在 ClearEngine 上测 11.residual 的值**：确认 residual 的有效值是否和 GPU 一致（~1e4），
   还是 ClearEngine 上小很多。如果 ClearEngine 的 residual 也是 ~1e4，那 values mask 的
   预除就不是原因。
2. **检查 values mask 的 FHE 执行**：mask 的编码和乘法是否正确？是否有 slot 出错？
3. **检查 `_fold_into_slot_zero`**：rotate+sum 累积是否在 GPU 上引入了额外误差？
4. **对比 11a1/11a2 的 p50**：GPU 的 p50 是 1.895e-08（11a1）和 1.049e-12（11a2），
   而 max 是 4.76e7 和 2.34e5。**p50 极小但 max 极大**——说明只有少数 slot 炸了，
   大部分 slot 是对的。这指向 FHE 噪声在少数 slot 上的异常放大。

---

## 6. 请协作者确认

1. ClearEngine 上 11.residual 的 max 值是多少？和 GPU 的 9.348e+04 比如何？
2. 11a1 的 p50 = 1.895e-08（GPU）vs ClearEngine 的 p50 是多少？如果 ClearEngine p50
   也是 ~1e-8，那大部分 slot 是对的，只有少数 slot 炸了。
3. values mask 的预除是否在 FHE 下有已知问题？

Co-Authored-By: CodeAgent (GLM-5.2) <noreply@anthropic.com>
