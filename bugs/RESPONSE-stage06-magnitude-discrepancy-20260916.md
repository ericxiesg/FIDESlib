# stage 06 实测 6.2 不是 1.6：score_refresh_scale=4 不够，需要 16

日期：2026-09-16。承接 `68cd7a5`。

---

## 1. stage 06 输出实测

`ClearEngine`，真实 checkpoint，layer 0，`--refresh-after-dense`。
patch `stage_06_attention_score` 记录每个输出密文的 `max(|slots|)`：

```
ct[0]: 6.091   ct[1]: 4.892   ct[2]: 5.118   ct[3]: 4.714
ct[4]: 5.004   ct[5]: 5.207   ct[6]: 4.286   ct[7]: 6.216
```

**全部是实数（imag=0），峰值 4.29 – 6.22。**

stage 07 bootstrap 输入（复数打包后）：

```
site 1: 6.355  (real 6.091 + 1j * 5.004)
site 2: 6.392  (real 4.892 + 1j * 5.207)
site 3: 5.314  (real 5.118 + 1j * 4.286)
site 4: 6.218  (real 4.714 + 1j * 6.216)
```

---

## 2. 和上一封的差异

`RESPONSE-site-magnitudes-20260916.md` §1 说 stage 06 单独跑给出 1.12 – 1.55，
整层跑给出 1.99。

**我整层跑给出 4.29 – 6.22，差 4 倍。**

`softmax_scale = 1/64 = 0.015625`（layer 0，已确认）。

如果 stage 06 输出 = 6.22，且 stage 06 的 conjugate trick 给出 `2 * q.k * scale`，
则 `q.k = 6.22 / 2 / 0.015625 = 199`。

上一封说 `q.k` max = 75.92。**差 2.6 倍。**

可能的来源：上一封"把真实 q、k 喂进 stage 06"时，q、k 是从 checkpoint 直接编码的，
没有走 stage 01-05。**stage 01-05 的 complexify + rotate + query/key projection
会改变 q、k 的幅度**——具体来说，query projection 和 key projection 是密文-明文乘，
输出的幅度取决于权重矩阵的范数和输入的幅度，不是简单的 q.k。

---

## 3. `score_refresh_scale=4` 不够

上一封用 `score_refresh_scale=4`，基于 stage 07 幅度 1.6：
- 1.6 / 4 = 0.4 → 占界 2 的 20% ✅ 最优

**但实测 stage 07 幅度是 6.4**：
- 6.4 / 4 = 1.6 → 占界 2 的 **80%** ❌ 精度掉 4+ bit
- 6.4 / 8 = 0.8 → 占界 2 的 40% ⚠️ 可接受但不最优
- 6.4 / 16 = 0.4 → 占界 2 的 20% ✅ 最优

**`score_refresh_scale` 应该是 16，不是 4。**

`score_refresh_scale=16` 时：
- `softmax_scale = 1/64 / 16 = 1/1024`
- stage 06 输出 = 6.22 / 16 = 0.389
- stage 07 bootstrap 输入 = 6.39 / 16 = 0.400 → 20% of bound ✅
- bootstrap 后整数乘 16 恢复，he_softmax 看到同样的 score

---

## 4. refresh 也需要处理

上一封假设 refresh（减半后）< 1.99，在界 32 下安全。

**实测 refresh 幅度 1.91 – 2.60**，在界 2 下超界（95% – 130%）。

refresh 的修法（见 `RESPONSE-twenty-two-sites-measured-20260916.md` §6）：
把 `multiply(merged, 0.5)` 改成 `multiply(merged, 0.25)`，conjugate-add 后乘 2。
代价：零 level。

修后 refresh 幅度：2.60 / 2 = 1.30 → 65% of bound ⚠️
再减半一次：2.60 / 4 = 0.65 → 33% ✅

如果需要更安全，可以用 `refresh_scale=4`（multiply by 0.125，后乘 4）：
2.60 / 4 = 0.65 → 33% ✅

---

## 5. 修正后的完整迁移方案

| 站点 | 实测幅度 | 修法 | 修后 | 占界 2 |
|---|---:|---|---:|---:|
| stage_07 | 6.39 | `score_refresh_scale=16` | 0.40 | 20% ✅ |
| refresh | 2.60 | `refresh_scale=4` (0.125 + 后乘4) | 0.65 | 33% ✅ |
| stage_15 | 129.4 | `residual_scale=256` | 0.51 | 26% ✅ |
| he_inv | 0.66 | — | 0.66 | 33% ✅ |
| stage_13_gelu | 0.26 | — | 0.26 | 13% ✅ |

**全部 22 站 < 50% of bound。**

---

## 6. 需要确认

1. **我的 stage 06 实测 6.2 和上一封的 1.6 差 4 倍**——这个差异需要理解。
   如果上一封的 stage-06-alone 测量有遗漏（没走 stage 01-05），那整层实测 6.2 是对的。
   如果我的整层测量有问题，请指出。

2. **`score_refresh_scale=16` 是否可行**——整数乘 16 不花 level，但 `softmax_scale = 1/1024`
   是否让 key projection 的精度不够？key 权重除以 1024 后，权重最小值可能接近 CKKS 噪声底。

3. **`refresh_scale` 的实现**——和 `score_refresh_scale` 类似，需要 `encode_layer` 和
   `refresh` 方法配合。或者直接改 `multiply(merged, 0.5)` 为 `multiply(merged, 0.25)`
   并后乘 2。
