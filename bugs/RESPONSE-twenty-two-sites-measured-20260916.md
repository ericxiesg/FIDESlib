# 22 个 bootstrap 站点幅度实测：发现第三处超界（refresh）

日期：2026-09-16。承接 `395379d`。

---

## 1. 方法

`ClearEngine`，真实 checkpoint，layer 0，`--refresh-after-dense`。

monkey-patch `ClearEngine.bootstrap`：记录调用栈和输入幅度（`max(|slots|)`），
返回 level=999 的密文（让计算走完所有 22 站不 crash）。
同时 patch 掉 `he_inv` 的 range check，让 softmax 的 `he_inv` 不中途报错。

禁用 `skip_pointless_bootstraps`（直接调 `engine.bootstrap`），确保每一站都被记录。

---

## 2. 22 个数

| # | 站点 | 代码位置 | 幅度 | in_level |
|---|---|---|---|---|
| 1 | stage_07_softmax | softmax.py:178 | 6.355 | 25 |
| 2 | stage_07_softmax | softmax.py:178 | 6.392 | 25 |
| 3 | stage_07_softmax | softmax.py:178 | 5.314 | 25 |
| 4 | stage_07_softmax | softmax.py:178 | 6.218 | 25 |
| 5 | he_inv (he_softmax) | numeric.py:190 | 0.0006 | 988 |
| 6 | he_inv (update_inv_D) | numeric.py:190 | 0.657 | 26 |
| 7 | **refresh** | **layernorm.py:178** | **1.905** | 19 |
| 8 | **refresh** | **layernorm.py:178** | **2.317** | 19 |
| 9 | **refresh** | **layernorm.py:178** | **2.017** | 19 |
| 10 | **refresh** | **layernorm.py:178** | **2.602** | 19 |
| 11 | stage_13_gelu | feedforward.py:86 | 0.199 | 18 |
| 12 | stage_13_gelu | feedforward.py:86 | 0.262 | 18 |
| 13 | stage_13_gelu | feedforward.py:86 | 0.220 | 18 |
| 14 | stage_13_gelu | feedforward.py:86 | 0.232 | 18 |
| 15 | stage_13_gelu | feedforward.py:86 | 0.236 | 18 |
| 16 | stage_13_gelu | feedforward.py:86 | 0.228 | 18 |
| 17 | stage_13_gelu | feedforward.py:86 | 0.208 | 18 |
| 18 | stage_13_gelu | feedforward.py:86 | 0.248 | 18 |
| 19 | stage_15_prepare_layernorm | layernorm.py:195 | 93.55 | 22 |
| 20 | stage_15_prepare_layernorm | layernorm.py:195 | 94.37 | 22 |
| 21 | stage_15_prepare_layernorm | layernorm.py:195 | 100.93 | 22 |
| 22 | stage_15_prepare_layernorm | layernorm.py:195 | 129.36 | 22 |

按站点分组：

| 站点 | 次数 | 幅度范围 |
|---|---|---|
| stage_07_softmax | 4 | 5.31 – 6.39 |
| he_inv | 2 | 0.0006 – 0.66 |
| **refresh** | **4** | **1.91 – 2.60** |
| stage_13_gelu | 8 | 0.20 – 0.26 |
| stage_15_prepare_layernorm | 4 | 93.6 – 129.4 |

---

## 3. 三处超界，不是两处

在 q0/Delta=2（迁移目标）下：

| 站点 | 幅度 | 占界 | 状态 |
|---|---|---|---|
| stage_07_softmax | 5.3 – 6.4 | 265% – 320% | ❌ 超界 |
| **refresh** | **1.9 – 2.6** | **95% – 130%** | **❌ 超界** |
| stage_15_prepare_layernorm | 93.6 – 129.4 | 4680% – 6470% | ❌ 超界 |

**`refresh`（layernorm.py:178）是第三处超界**，之前没有被发现。

`refresh` 虽然在 bootstrap 前减半（`multiply(merged, 0.5)`），但 attention dense 的
输出是 3.8 – 5.2，减半后 1.9 – 2.6，**仍然超过界 2**。

在 q0/Delta=32（当前 EasyFHE 参数）下，refresh 占 6% – 8%，不起眼——
和 stage 07 一样，在 32 的界下完全安全，**只在界变成 2 之后才暴露**。

---

## 4. 关于 §2 的 2.4 倍差异

上一封（`RESPONSE-stage07-also-over-20260916.md`）§2 说 stage 07 算出来 4.745
但实测 1.99，差 2.4 倍。

**我这边实测 stage 07 是 5.3 – 6.4**，和算出来的 4.745（复数打包上界 6.711）一致，
**不是 1.99**。

而 1.91 – 2.60 这个范围出现在 **refresh** 站点（layernorm.py:178），不是 stage 07。

可能的解释：上一封的"实测 1.99"量到的是 refresh 而不是 stage 07。
两个站点的 call chain 不同（`softmax.py:178` vs `layernorm.py:178`），
但如果只看幅度不看调用栈，1.99 和 1.91 很接近。

---

## 5. 在 q0/Delta=32 下

| 站点 | 幅度 | 占界 | 状态 |
|---|---|---|---|
| stage_07_softmax | 6.4 | 20% | ✅ |
| refresh | 2.6 | 8% | ✅ |
| stage_15 | 129.4 | 404% | ❌ → residual_scale=256 → 0.50 → 1.6% ✅ |
| he_inv | 0.66 | 2% | ✅ |
| stage_13_gelu | 0.26 | 0.8% | ✅ |

**在 q0/Delta=32 + residual_scale=256 下，全部 22 站安全。**

---

## 6. 在 q0/Delta=2 下（迁移目标）

| 站点 | 幅度 | 占界 | 修法 | 修后 |
|---|---|---|---|---|
| stage_07 | 6.4 | 320% | `score_refresh_scale` (已实现) | 6.4/s → 需 s≥4 |
| **refresh** | **2.6** | **130%** | **？** | **需缩小到 <2** |
| stage_15 | 129.4 | 6470% | `residual_scale=256` (已实现) | 0.50 → 25% ✅ |
| he_inv | 0.66 | 33% | — | ✅ |
| stage_13_gelu | 0.26 | 13% | — | ✅ |

**三处需要处理，不是两处。** `score_refresh_scale` 和 `residual_scale` 已经覆盖
stage 07 和 stage 15，但 **refresh 还没有修法**。

### refresh 的可能修法

`refresh` 已经有 `multiply(merged, 0.5)`。把 0.5 改成 0.25（多减半一次），
然后在 conjugate-add 之后 multiply by 2（整数乘，不花 level）：

```python
# 现在：
merged = self.bootstrap(self.rescale(self.multiply(merged, 0.5)))
# = 0.5 * input → bootstrap → 2 * output = 1.0 * input  (identity)

# 改后：
merged = self.bootstrap(self.rescale(self.multiply(merged, 0.25)))
# = 0.25 * input → bootstrap → 2 * output = 0.5 * input
# 然后 multiply by 2 → 1.0 * input  (identity, integer multiply = no level)
```

代价：无（0.25 和 0.5 一样是 fractional multiply + rescale，多出的 2 是整数乘）。

---

## 7. `score_refresh_scale` 需要多少

stage 07 幅度 max=6.39。要降到 q0/Delta=2 的 50%（=1.0）以留余量：

```
6.39 / s ≤ 1.0  →  s ≥ 7
```

要降到 25%（=0.5）：

```
6.39 / s ≤ 0.5  →  s ≥ 13
```

`s=8` 给 0.80（40%），`s=16` 给 0.40（20%）。

`score_refresh_scale` 必须是整数（stage 07 用整数乘恢复），所以 `s=8` 或 `s=16`。

---

## 8. 下一步

1. **确认 refresh 修法**：把 `multiply(merged, 0.5)` 改成 `multiply(merged, 0.25)` + 后乘 2，
   在 ClearEngine 上验证仍然是 identity。
2. **定 `score_refresh_scale`**：s=8 还是 s=16，取决于 he_exp 多项式在减半后的精度。
   上一封 §4 说"减半+多平方一次"不行（Goldschmidt 9→19 迭代），但 `score_refresh_scale`
   不改 he_softmax 的输入，所以这个问题不存在——它只改 bootstrap 的输入幅度。
3. **跑 ClearEngine 全量**：s=8 + refresh 0.25 + residual_scale=256，验证 22 站全部 < 1.0。
4. **迁移到 sb=59/fmb=60**：然后跑 GPU 端到端。
