# 更正：binary 只省 1.0 GiB 不是 1.8；而缺口正好是 3 个 level

日期：2026-09-18。更正 `81108ad` 和 `7d227c4` 里的一个数。

---

## 0. 先更正

我之前说"退回 binary 旋转密钥省 **1.8 GiB**"。**那个数是错的。**

1.8 是拿 `--keys 21` 和 `--keys 15` 比出来的，而 `--keys N` 假设**密钥不截断**。
真实的计划是**按 level 截断**的。用真计划（`budget` 不带 `--keys`，跑干运行）重新量：

| depth 41 | rotation keys | 合计 | 余量 |
|---|---|---|---|
| binary（15 把） | 2.8 GiB | **30.1 GiB** | 1.9 GiB |
| +6 factored（21 把） | 3.8 GiB | 31.1 GiB | 0.9 GiB |

**binary 省的是 1.0 GiB，不是 1.8。** 结论也跟着变：**binary 单独不够**。

---

## 1. 卡能装到 depth 38；整模型要 depth 41

真计划 + binary，逐 depth：

| depth | 合计 | 余量 | 够 keygen 的 3 GiB 吗 |
|---|---|---|---|
| 37 | 28.3 GiB | 3.7 GiB | ✅ |
| **38** | **28.8 GiB** | **3.2 GiB** | ✅ |
| 39 | 29.2 GiB | 2.8 GiB | ✗ |
| 40 | 29.7 GiB | 2.3 GiB | ✗ |
| 41 | 30.1 GiB | 1.9 GiB | ✗ |

**depth 38 是这张卡能撑的最深。整模型实测要 depth 41。缺 3 个 level。**

---

## 2. 而 `he_inv` 的 epsilon 正好值 3 个 level

`7d227c4` 里我量过：把 epsilon 收到实测比值（margin 2），
`he_inv` 十二层 166 次迭代 → 130 次，**每层省 3.0 个 level**。

**3 = 3。** 这不是"锦上添花的 3 个 level"，这是**决定整模型装不装得下的那 3 个**。

所以那条我当时按"证据不够"搁置的工作，现在是**关键路径**。它值得按正经做法做完：

- 要么重做 `precision / 128 / 2` 的推导（128 是 token 数，形状是"和已知时平方和的下界"那一类），
  看真实的界能收到多紧；
- 要么在 FHE 上拿**很多句话**重新标定，而不是我手上的一句。

两条都不是拍脑袋能定的——越界的后果是 Goldschmidt **静默饱和**（返回有限的、看起来合理的错数），
而设备上没有量程检查。

---

## 3. 现在的完整账（全部真计划实测）

```
整模型需要        depth 41
卡能装到          depth 38（binary，余量 3.2 GiB ≥ keygen 3 GiB）
缺口              3 个 level
he_inv epsilon    正好 3 个 level/层
```

其它可省 level 的手段（已做的）：每层方差窗口 **4 个/层**（`42a43ed`），
层边界 refresh（`2e78bcd`，让 12 层跑得起来）。
已否决的：`--inverse-lift` 当 level 用（0.2 个/层）。

---

## 4. 对你没影响

你在跑单层 depth 37，binary 在 depth 37 有 3.7 GiB 余量，本来就够。
这份是给"整模型什么时候能上卡"定的量。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
