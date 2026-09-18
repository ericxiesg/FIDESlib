# 一层吃掉 36 个 level，层与层之间没有自举——**所以这个项目从来只跑过 1 层**

日期：2026-09-18。承接 `4a849b1`。

---

## 0. 结论

我想量每层的分母上界（给 `--inverse-lift` 定每层的安全值），于是第一次跑了 `--layers 12`。
**第 1 层直接死了。**

```
LAYER 0: in level 37 -> out level 1, |x| max 19.44, 8 ct
ScaleMismatch: level_down(by=1): would leave the ciphertext at level -1
```

**一层吃掉 36 个 level**（进 37，出 1），而层与层之间**没有任何自举**。
第 1 层一上来就要 20 个 level，手里只有 1。

之所以一直没人碰到：**所有跑法都是 `--layers 1`。** 你的、我的、报告里的，全是。

---

## 1. 修法：层边界加一次 refresh

`--boundary-refresh-scale S`（默认 **4**，0 表示关掉）。只对第 1 层及之后生效，第 0 层是新鲜密文。

不能直接自举：状态离开一层时 **|x| max 19.44**，而 q0/(2Δ) = **16**——**已经越界了**，
而且还在长（第 1 层出来是 22.62）。所以先除。`refresh` 内部本来就有一次折半，
再除一个整数既不花 level 也不花 scale degree。

### 实测

```
LAYER 0: in level 37 -> out level 1, |x| max 19.44
LAYER 1: in level 20 -> out level 1, |x| max 22.62

hidden after layer 1  MAE 2.026e-03  relRMSE 3.612e-03   label agreement 100%
```

（layer 0 单独跑是 2.776e-03，两层是 3.612e-03——误差按预期缓慢累积。）

代价：每个层边界 4 次 bootstrap。按 `--time-ops` 的 189 ms/次 = **0.76 s/层**。

---

## 2. 修完之后，下一个拦路的是**标定**，不是算术

带上边界 refresh 再跑 12 层，跑到**第 1 层的 LayerNorm** 才停：

```
ValueError: he_invsqrt: the variance runs over [0.02823, 2.498] (median 0.06894)
on the 2048 slots that carry a statistic, but the iteration is set up for [0.001333, 1].
```

`LayerNormStages.VARIANT_BOUNDS = {2: (1e-5, 0.2, 150.0), 3: (1e-5, 0.75, 2500.0)}`
——**按 variant 分，不按层分**，而且显然是照着第 0 层定的。第 1 层的方差超出 2.5 倍。

`Softmax.LAYERS` 是有每层条目的（`shift`、`inv_epsilon`），`VARIANT_BOUNDS` 没有。
**多层推理的下一步是把方差窗口也做成每层标定。** 我没有动它——
窗口的*比值*决定 `he_invsqrt` 的迭代次数也就是 level 开销，随便放宽会连带改 level 预算，
这需要按层量过再定，不是猜的。

---

## 3. 顺带：`--inverse-lift 3` 只对第 0 层安全

同样是这次 12 层跑，带 lift 3 时另一层报：

```
ValueError: he_inv: the denominator runs over [0.06966, 2.042] ... set up for [0.01167, 1]
```

**量程检查按预期拦住了。** 这正是我上一份报告里说的"安全的 lift 每层不同，只量过第 0 层"。
所以：**`--inverse-lift 3` 现在只能用于 `--layers 1`**，多层需要每层的值。

---

## 4. 对你当前工作的影响

**基本没有**——你在跑的是 `--layers 1`，这条路径一个字节都没变（边界 refresh 只在
`layer_index > 0` 触发）。

但它改变了"还差多远"的判断：之前的隐含假设是"精度修好就能跑整个模型"，
实际上**整个模型从来没跑过**，而且中间还隔着每层的方差标定。
所以路线是：

1. 精度（`he_inv` 发散）← 你我都在这
2. 边界 refresh ← 本次，已修
3. 每层方差窗口标定 ← 新发现的下一个
4. 每层 `inverse_lift` ← 可选，第 0 层实测 3–8 倍

217 passed, 67 skipped。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
