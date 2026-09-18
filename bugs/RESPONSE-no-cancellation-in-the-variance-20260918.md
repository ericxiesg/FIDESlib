# 不是相消：两项相差 122 倍，条件数 1.008

日期：2026-09-18。针对 `f63b733`。

---

## 0. 结论

我加了 `11a1` / `11a2` 两个探针，把相减前的两项各自量出来了。ClearEngine，真实 checkpoint：

```
[probe] 11a1.n_sum_of_squares   level 17  max +0.2161      p99 0.1579
[probe] 11a2.squared_total      level 17  max +0.001773    p99 0.001637
[probe] 11a.variance            level 17  max +0.2161      p99 0.1578
```

`(Σx)²` 是 `n·Σx²` 的 **1/122**。相减只拿掉 0.8%。**条件数 1.008，不是灾难性相消。**
stage 16 那次同理（0.1765 vs 0.00025，1/706）。

原因是 `values` 掩码已经先除掉了 `sqrt(max_denominator) = sqrt((1.05·max_var + var_e)·n²)`，
而 LayerNorm 的输入均值本来就接近 0，所以 `(Σx)²` 项一开始就很小。
我之前说的"条件数 1.01"就是这个，现在是量出来的而不是推出来的。

---

## 1. 还有一个更硬的理由

**相消只会让结果变小，不会变大。**

`n·Σx²` 和 `(Σx)²` 在 clear 上的最大值是 **0.216**。你在 GPU 上量到的 `11a.variance`
是 **4.628e7**——比两个被减数**都大 2.1 亿倍**。

任何两个 0.2 量级的数相减都不可能得到 4.6e7。所以这个数不是相减产生的，
**是相减之前某一项就已经错了 8 个数量级**。

---

## 2. 请按这个顺序量（探针都已经在了）

按 commit `a22ec04`，`--per-stage` 现在会打出：

```
10.attention_dense      clear: level 1   max +5.23
11.residual             clear: level 20  max +10.76    (min -26.72)
11a1.n_sum_of_squares   clear: level 17  max +0.2161
11a2.squared_total      clear: level 17  max +0.001773
11a.variance            clear: level 17  max +0.2161
```

从上往下找第一个对不上的：

- **`11.residual` 就炸了** → 问题在 stage 10 或 residual 那条 skip，和 LayerNorm 无关。
- **`11.residual` 正常、`11a1` 炸了** → 在 `masked` / `square` / `_fold_into_slot_zero` /
  `statistic` 这一段，是 `values` 缩放或者 interval_sum 的问题。
- **两项都正常、`variance` 炸了** → 那才轮到 `subtract`，那时我们再看 scale/noise_level。

2.14e8 这个倍数本身也是线索：它不是 Δ=2^50，也不是 `sqrt(max_denominator)=2489` 或
它的平方 6.19e6，所以不像单一一次缩放丢失。

---

## 3. 顺带排除：整数标量乘法两边一致

`variance = subtract(multiply(sum_of_squares, n), squared_total)` 里的 `n` 是 Python int。

- `ClearEngine.multiply`（`clear.py:269`）："Integer scalars are level- and scale-free"
- `pyfideslib.Engine.multiply`（`__init__.py:164`）：`EvalMultByInteger`，同样 level-free
- `Stages.plaintext` 对非 ndarray 原样返回，所以 int 不会被转成 float 走 `EvalMultScalar`

两边模型一致，**不是这里**。（如果 `n` 曾经是 float，它会走 `EvalMultScalar`、吃一层、
把密文留在 Δ²，然后和 Δ 的 `squared_total` 相减——那才会出你看到的那种量级。现在不是。）

---

## 4. 关于修法 A / B

在拿到 §2 的结果之前不要动算法。

- **A（改成 `Σ(x−mean)²`）**：现在没有理由做。条件数 1.008，改成中心化版本要多一轮
  broadcast + 一层 level，买不到精度。等 §2 证明确实是相消再说。
- **B（Meta-BTS 提精度）**：`bootstrap_twice` 已经在 `stages.py` 里（`834c73b`），
  k=10。但它治的是 bootstrap 精度，不治一个已经错了 8 个数量级的中间值。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
