# 试了把 `/128` 换成真实 token 数——**不安全，已回退**

日期：2026-09-21。负面结果一则，**结论是那 3 个 level 拿不到**。

---

## 0. 背景

整模型要 depth 41，卡只装得下 depth 38，**差 3 个 level**（`a49bae5`）。
而 `he_inv` 的 epsilon 正好值 3 个 level/层。我之前按"证据不够"搁置了它。

这次我以为找到了一个**不需要赌数据**的收紧方式。试了，**错的**。

---

## 1. 我的想法（听起来很有道理）

`update_inv_D` 里：

```python
epsilon = precision / 128 / 2
```

128 是 `g.dim`，也就是**满序列**的 token 数。而那个界的形状是
"n 个数之和为 S，平方和至少 S²/n"，等号在 n 个数全相等时取到（均匀注意力）。

关键是：分子在平方**之前**已经被 `attention_mask` 乘过了，所以 padding 的 key
贡献**精确为 0**——求和实际只有 `real_tokens` 项，不是 128 项。

所以用真实 token 数才是**同一个界的正确 n**，不是对数据的赌。MRPC 的 44 token 行上
是 128/44 = 2.9 倍更紧，正好**一次 Goldschmidt 迭代 = 一个 level**。

---

## 2. 实测：不成立

改完跑测试，`_check_inversion_range` **当场拦下**：

```
ValueError: he_inv: the denominator runs over [0.02052, 0.05072] (median 0.03057)
on the 24576 slots that carry data, but the iteration is set up for [0.04161, 1].
```

收紧后的 epsilon 是 **0.04161**，而实际到达的分母最小值是 **0.02052**——**在界下面**。

所以那个 128 **不是"求和有多少项"**。它编码的是别的东西（`precision`、`k` 的缩放、
那次加倍之间的关系），我那套重新解释经不起测量。

**已回退**，并把这段写进了 `softmax.py` 的注释，免得下一个人（包括我自己）再走一遍。

---

## 3. 这个负面结果本身是有用的

1. **它加强了原来的判断。** 我之前说"那个界是推导出来的，不能拿一句话的观测去收紧"。
   现在连一个看起来**不依赖数据**的收紧都被实测否决了。那 3 个 level 比我想的更难拿。

2. **护栏起作用了。** `_check_inversion_range` 在 ClearEngine 上**立刻**拦住了，
   没有让它变成"跑完之后数字有点不对"。这正是这些量程检查存在的理由——
   设备上没有这道检查，如果这个改动被推到设备，表现会是 Goldschmidt **静默饱和**，
   返回一个有限的、看起来合理的错数。

3. **整模型装卡这条路要另找。** 现在的账：

   | | |
   |---|---|
   | 整模型需要 | depth 41 |
   | 卡装得下 | depth 38（binary，余量 3.2 GiB ≥ keygen 3 GiB） |
   | 缺口 | 3 个 level |
   | `he_inv` epsilon | 值 3 个 level，**但两种收紧方式都被否了** |
   | 已拿到的 | 每层方差窗口 4 个/层（`42a43ed`） |

   下一个候选是"融合 stage 03/04/05"（省的是显存不是 level，约 2 GiB），
   或者重做 `/128` 那个推导——但那需要真正读懂它编码的是什么，不是猜。

221 passed, 67 skipped。**不影响你手上的单层工作。**

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
