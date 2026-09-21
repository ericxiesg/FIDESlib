# 发散的起点在 `07a`：**分数跑出了 he_exp 的拟合窗口**

日期：2026-09-21。针对 `e853110` 和 `3775678`。

---

## 0. 先说好消息：**第一次 he_inv 已经修好了**

你的逐槽数据里最重要的一行：

```
07.inv_iter01_b  min +0.023  max +1.529  <0 0/32768  >1 1/32768
```

`min` 是**正的**，`<0` 是 **0**。PADDING_FLOOR 完全生效，第一次 he_inv 干净收敛。
负分母那条线可以结掉了。

---

## 1. 但真正的起点比 he_inv 早两步

```
07a.refreshed_scores   设备 min -16.05  max +37.5     本机 min -10.41  max +12.43
```

**layer 0 的 he_exp 拟合窗口是 `[-27.2493, 21.7269]`。**

**37.5 > 21.73。** 分数跑到窗口外 **1.73 倍**。

minimax 拟合只在它的区间上是拟合。出了区间，degree-15 多项式**不是变差，是起飞**。
后面每一条都是这个的下游：

| | 设备 | 本机 | |
|---|---|---|---|
| `07b.exp` | `<0 86546` | exp 不可能是负的 | ← 多项式在窗口外 |
| `07c.denominator` | `<0 12080` | 平方和不可能是负的 | ← 同上 |
| `07d` | 5.7e5 | 0.1475 | ← he_inv 拿到坏输入 |
| 第二次 he_inv `iter01_b` | 1e6，半数槽 `<0` | — | ← `update_inv_D` 把坏槽平方了 |

你问"为什么第二次 he_inv 的 `a` 在 iter01 就到 ±2000"——因为它的输入本来就是坏的，
不是它自己发散。**he_inv 一直是受害者。**

### 已加的护栏

`he_exp` 之前是全仓库唯一**没有量程检查**的迭代数值（`he_inv`、`he_invsqrt` 都有）。
现在有了。本机实测它不误报：

```
[range] he_exp observed [-4.86508, 12.1823] against window [-27.2493, 21.7269]
```

设备上它**不会触发**（只有 ClearEngine 是 `inspectable`），所以设备靠探针——见下。

---

## 2. 分数是在哪被放大的？新探针 `07a0` 会告诉你

你的 per-stage 表里 `scores` 是**对的**（relRMSE 2.874e-07），而 `07a` 的 max 是本机的 3 倍。
中间只有一件事：**score refresh 的那次 bootstrap**。

新增 `07a0.score_refresh_input`——那次 bootstrap 的**输入**。本机基线：

```
07a0.score_refresh_input  4 ct  level 25  min -0.3199  max +0.3807
07a.refreshed_scores      8 ct  level 17  min -10.41   max +12.43
```

0.3807 × 2（fold 加倍）× 16（`score_refresh_scale` 还原）= 12.18 ✓ 对得上。

**请带 `--per-stage` 重跑，把 `07a0` 贴回来。** 两种结果，两条路：

- **`07a0` ≈ 0.38** → 输入是对的，是**那次 bootstrap 把它放大了 3 倍** → bootstrap 精度问题
- **`07a0` ≈ 1.17** → 输入就已经大了 → 问题在 stage 06 的打包，尽管 `scores` 比对是准的

---

## 3. 顺带发现一个我自己设错的参数：`--score-refresh-scale 16` 太大了

那次 bootstrap 的输入只有 **0.38**，而 `q0/(2Δ) = 16`——**只用了量程的 2.4%**。

bootstrap 的绝对误差**不随消息变小**（实测 0.017）。所以在 0.38 上做 bootstrap，
相对误差是 **4%**；把消息放大回去，误差也跟着放大。
`score_refresh_scale` 本来是为了不越界，但 16 远远超过了需要。

用我之前那个 headroom 工具量了一遍（真实 checkpoint）：

| `--score-refresh-scale` | bootstrap 峰值 | 占界 | 余量 | 相对精度（对比 16） |
|---|---|---|---|---|
| 1 | 6.39 | 40.0% | **2.50×** ← 太紧 | 16× |
| **2** | **3.20** | **20.0%** | **5.01×** | **8×** |
| 4 | 1.52 | 9.5% | ~10× | 4× |
| 16（现在） | 0.38 | 2.4% | 42× | 1× |

**建议 `--score-refresh-scale 2`**：相对精度比现在好 **8 倍**，余量仍有 5 倍。
1 能到 16 倍但余量只剩 2.5 倍，低于我那个工具自己给的"3 倍以下就该往下调"的线，不建议。

本机三个 scale 的 `relRMSE` **完全相同**（8.479e-04）——缩放是精确补偿的，
所以这在精确算术上是免费的，只在设备上有区别。

---

## 4. 请跑这一条

```bash
THORFHE_DEBUG=1 python3 -m thorfhe.bench fhe --engine fideslib --device cuda:0 \
  --depth 37 --dnum 4 --bootstrap-level-budget 3,3 --binary-rotations \
  --refresh-after-dense --residual-scale 256 --refresh-scale 4 \
  --score-refresh-scale 2 --layers 1 --limit 1 --compact --per-stage \
  --extra-rotation-keys 6 --rotation-max-steps 4 --inverse-lift 3 \
  --plaintext-cache <你的 cache 目录>
```

要三样：`07a0` 的 min/max、`07a` 的 min/max、以及 `07a` 的 max 有没有掉回 21.73 以内。

⚠️ 换了 `--score-refresh-scale` 之后**明文缓存的 tag 会变**（scale 是烤进权重的），
所以这次是冷跑，会重编一次。这是对的行为，不是 bug。

221 passed, 67 skipped。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
