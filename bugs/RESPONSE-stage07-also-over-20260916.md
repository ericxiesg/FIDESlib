# 那个 1.99 是 stage 07，而且模式很干净：不减半的两处，正是超界的两处

日期：2026-09-16。承接 `b9130fd`。

---

## 1. 站点映射（读代码，不用跑）

一层 22 次 bootstrap 分布在五处，其中**只有两处不在 bootstrap 前减半**：

| 站点 | 减半 | 内容 |
|---|---|---|
| `feedforward.py:86`（stage 13 前的刷新） | ✅ `multiply(merged, 0.5)` | intermediate |
| `layernorm.py` 的 `refresh`（stage 10.5） | ✅ `multiply(merged, 0.5)` | attention dense |
| `numeric.py:190`（`he_inv` 内） | — | 分母，本来就 ~1e-4 |
| **`softmax.py:178`（stage 07）** | ❌ **无** | 复数化的 score |
| **`layernorm.py:172`（stage 15）** | ❌ **无**（文档说是有意的） | 残差 |

而真实 checkpoint 上超界的正是这两处：**1.99 和 132**。

**所以那个 1.99 是 `softmax.py:178`**，不需要再跑一次去认领它。

---

## 2. 但幅度我算出来比实测大 2.4 倍，这个差要留着

明文 numpy，真实 checkpoint，layer 0：

```
q.k (未缩放)              max 75.92    p99 50.63
4*(q.k)*softmax_scale     max  4.745   p99  3.164      (softmax_scale = 1/64)
复数打包后（上界 sqrt2 倍）  max  6.711
```

`4 * (q.k) * scale` 这个式子来自 `layer.py:36` 的注释
（"measured end to end, `he_softmax` sees `4 * (q.k) * scale`"）。

**但实测的 bootstrap 输入是 1.99，不是 4.7。差 2.4 倍。**

我没有能解释它的模型。可能是那条注释里的 `4` 不对，可能是打包把两个对角线配对
的方式让最大值不共位，也可能 stage 06 的输出还有一层我没算进去的缩放。

**这个差本身要查**，因为迁移后要按幅度定缩放系数，而现在两个数差 2.4 倍。
不过**两个数都超过界 2**（1.99 是 99.5%，4.745 是 237%），所以结论不变。

---

## 3. 结论：迁移要处理的是两处，不是一处

```
stage 15 (layernorm.py:172):  132   -> residual_scale=256 -> 0.516  ✅ 已实现
stage 07 (softmax.py:178):   1.99   -> 还没有办法            ❌
```

---

## 4. stage 07 的修法比 stage 15 难，因为 softmax 不是尺度不变的

stage 15 能免费缩放，是因为它的下游是 LayerNorm，**尺度不变**。
stage 07 的下游是 softmax，**不是**——`layer.py:30` 的注释就写着
"`he_softmax` must be the BERT attention score itself, because a softmax is not scale-invariant"。

所以不能简单地把 score 缩小。但有一条路，而且是 THOR 自己在用的机制：

### 减半 + 多平方一次

`stage_07_softmax` 现在是：

```python
merged = add(scores[i], multiply_1j(scores[i + half]))
merged = self.bootstrap(merged)                    # <- 没有减半
...
refreshed[i] = add(merged, conjugated)             # = 2a，这就是"加倍"的来源
```

改成 `bootstrap(rescale(multiply(merged, 0.5)))` 之后，输出是 `a` 而不是 `2a`，
即 score 减半——也就是**温度加倍**。

而 `he_softmax` 有现成的温度旋钮：`n`。

```python
for _ in range(int(np.log2(n))):
    numerators = numerators ** 2        # 每平方一次，温度减半
```

**`n` 从 2 改成 4 多平方一次，正好把温度补回来。** 这和 `update_inv_D`
（"one halving of the temperature"）是同一个机制。

代价是一次平方，**一个 level**。而我们的 level 预算在 depth 37 下余量是 1
（见 `RESPONSE-bootstrap-noise-level-20260915.md` 的实测）——**刚好够，没有第二次机会**。

### 需要先确认两件事

1. **`he_exp` 的多项式还准不准**：减半之后 `he_exp` 的输入范围减半，
   而 `min_x`/`max_x` 要跟着减半。但 `he_exp` 内部的 `scale = 64 if wide else 32`
   是**硬编码**的，不随范围走。范围减半而 scale 不变，多项式的有效精度会变——
   这要在 `ClearEngine` 上量，我可以做。
2. **多平方一次之后 `l` 和 `inv_epsilon` 要不要跟着调**：
   `update_inv_D` 的次数由 `l` 定，而分母会变。

---

## 5. 所以请把 22 个数按站点报出来

我上一封要的还是要——不是为了认领 1.99（那已经确定是 stage 07），
而是为了**看还有没有第三处**。

现在已知的只有"前五大"，那 17 个更小的在界 32 下都安全，
但**界变成 2 之后有没有第三处贴上来，我们没有数**。

这次已经吃过一次亏：最大值 132 之外冒出了一个 1.99，而它当时在 32 的界下
只占 6%，完全不起眼。

---

## 6. 我接下来做

第 4 节那两件事里的第一件——在 `ClearEngine` 上量"score 减半 + 多平方一次"
是否还能复现真 softmax，用 `test_stage9_thor_softmax.py` 已有的对照。
不需要 GPU，也不需要整层。
