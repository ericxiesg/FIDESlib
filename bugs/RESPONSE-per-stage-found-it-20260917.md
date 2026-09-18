# --per-stage 定位到发散点：norm_1（stage 11 attention layernorm）

日期：2026-09-17。针对 `13ff856`。

---

## 1. 逐 stage fidelity

```
stage                scale         MAE        RMSE       max       relRMSE    状态
query              2.0000       1.888e-07   2.641e-07   2.536e-06   2.884e-07   ✅
value              2.0000       1.204e-07   1.673e-07   1.390e-06   3.141e-07   ✅
scores             0.0313       3.763e-07   5.061e-07   3.695e-06   2.871e-07   ✅
softmax            1.0075       1.468e-05   3.442e-05   6.115e-04   5.645e-04   ✅
attention_dense    2.0147       2.025e-04   2.843e-04   2.556e-03   1.078e-03   ✅
norm_1            -1.5e+107     2.020e+02   2.534e+02   1.025e+03   1.967e+02   ❌ 炸
intermediate      -2.4e+49      9.414e+02   1.180e+03   5.944e+03   4.793e+02   ❌
gelu              -2.2e+128     8.194e+01   1.026e+02   4.908e+02   4.338e+02   ❌
output_dense       1.2e+95      6.037e+01   7.555e+01   2.970e+02   1.287e+02   ❌
norm_2             4.9e+17      9.832e+01   1.234e+02   5.461e+02   1.855e+02   ❌
```

**`norm_1`（stage 11，attention LayerNorm）是发散起点。**

- `attention_dense` 精确到 MAE=2e-4 ✅
- `norm_1` 直接炸到 scale=-1.5e+107, MAE=202 ❌

---

## 2. norm_1 = stage 11 = refresh + he_invsqrt + LayerNorm

`norm_1` 对应 `stage_11_attention_layernorm`，包含：
1. `refresh`（4 次 bootstrap，layernorm.py:178）
2. `he_invsqrt`（计算 1/sqrt(var)）
3. LayerNorm 的 multiply + subtract

`refresh` 和 `he_invsqrt` 都调用 `he_inv`，而 `he_inv` 内部调 bootstrap。
这是 22 站里 `he_inv` 那个 0.971（97% of recoverable bound）所在的位置。

---

## 3. 07a-07d 探针（协作者要的）

```
07a.refreshed_scores   8 ct  level 17  min -10.42  max +12.43  |x| p50 0.9275  p99 6.526
07b.exp                8 ct  level 9   min ~0      max +0.1055  |x| p50 ~0     p99 0.002541
07c.denominator        1 ct  level 9   min ~0      max +0.3037  |x| p50 ~0     p99 0.1747
07d.inverse_denom      1 ct  level 15  min ~0      max +0.1477  |x| p50 ~0     p99 0.1092
```

**softmax 链完全正常**——denominator 在 [0, 0.30]，inverse 在 [0, 0.15]，
都在 [eps, 1] 范围内。**softmax 不是问题。**

问题在 softmax 之后的 `norm_1`。

---

## 4. 结论

| 检查项 | 结果 |
|---|---|
| softmax 链（07a-07d） | ✅ 正常 |
| attention_dense | ✅ MAE=2e-4 |
| **norm_1（stage 11）** | **❌ scale=-1.5e107，发散起点** |
| bootstrap 单次精度 | 15.2 bit（够用） |
| noise model（bootstrap 误差） | ✅ 不炸 |

**发散在 `norm_1` = `refresh` + `he_invsqrt` + LayerNorm。**
不是 bootstrap 精度（15 bit 够），不是 softmax（07a-07d 正常）。

**最可能的原因**：
1. `he_invsqrt` 里的 `he_inv` 在 GPU 上的实际精度不够（不是 bootstrap 本身，
   而是 Goldschmidt 迭代在 GPU 上的累积舍入）
2. `refresh` 的 bootstrap 在链底（低 level）时精度下降
3. LayerNorm 的 `statistic_mask` 或 `VARIANT_BOUNDS` 在 GPU 上行为不同

---

## 5. 下一步

1. **在 `norm_1` 内部加探针**：分别测 refresh 输出、he_invsqrt 输出、LayerNorm 最终输出的精度
2. **测 `he_invsqrt` 单次精度**：和 bootstrap 一样，在 GPU 上直接测
3. **检查 `refresh` 的 bootstrap 输入 level**：per-stage 显示 softmax 在 level 17，
   attention_dense 应该在 level ~14，refresh 在 level ~11——低 level 的 bootstrap 可能精度更差
