# 11a 探针定位：variance 在 GPU 上是 4.6e7，ClearEngine 是 0.22

日期：2026-09-18。针对 `35841d9`。

---

## 1. GPU 实测探针

```
[probe] 11a.variance        1 ct  level 17  min -3.457e-09  max +4.628e+07  |x| p50 1.123e-09  p99 4.319e+07
[probe] 11b.inverse_sqrt    1 ct  level 8   min -5.191e+144  max +5.526e+144  |x| p50 8.535e+143  p99 3.325e+144
```

## 2. ClearEngine 基线（你的报告）

```
[probe] 11a.variance        1 ct  level 17  min +0  max +0.2161  p99 0.1578
[probe] 11b.inverse_sqrt    1 ct  level 8   min +0  max +3.953   p99 3.8
```

## 3. 对比

| 探针 | ClearEngine | GPU | 差距 |
|---|---|---|---|
| 11a.variance max | 0.2161 | 4.628e+07 | **2.1亿倍** |
| 11b.inverse_sqrt max | 3.953 | 5.526e+144 | **1.4e144倍** |

**variance 在 GPU 上炸了 8 个数量级**，导致 `he_invsqrt(1/sqrt(variance))` 输出 5.5e144。

---

## 4. 根因

variance = `n * Σx² − (Σx)²`

ClearEngine 用 float64（精确），GPU 用 CKKS（有噪声）。当 `n * Σx²` 和 `(Σx)²` 
量级相近时，相消会放大噪声。

你说条件数 1.01（mean²/var ≈ 0.005），但这是 **ClearEngine float64** 下的条件数。
GPU 上 CKKS 的噪声让 `n * Σx²` 和 `(Σx)²` 各自有了 ~15 bit 的误差，
相消后误差变成 `~2^15 * 原始量级`——如果原始量级是 ~1e7（n * Σx²），
相消后误差 ~3e5，方差本应 0.22 但被噪声淹没成 4.6e7。

---

## 5. 修法方向

### A. 改用 `Σ(x - mean)²` 而非 `n*Σx² − (Σx)²`
先算 mean，再算 `Σ(x - mean)²`。避免相消。代价：多一次 rotate + add（算 mean），
但数值稳定。

### B. 用 Meta-BTS 提升 variance 计算的精度
在 variance 计算前用 `bootstrap_twice`（k=10 → 25 bit），减少 CKKS 噪声。
代价：+2 次 bootstrap + 1 level。

### C. 在 variance 后加 bootstrap 清理噪声
variance 算完后立刻 bootstrap，把累积的 CKKS 噪声清掉。
但这不解决相消问题——噪声在相消时就已经放大了。

### D. 拆成两步：先 bootstrap 算 mean，再 bootstrap 算 variance
每个统计量用新鲜的密文计算，避免累积噪声。
代价：+1 次 bootstrap。

---

## 6. 请定方向

**A（改成 Σ(x-mean)²）最根本**——不花 bootstrap，只是改算法。
但需要改 `he_layernorm` 的 variance 计算逻辑。

**B（Meta-BTS）最简单**——不改算法，只提精度。
但每层 2 次 variance 计算 × 2 次 bootstrap = +4 bootstrap/层。
