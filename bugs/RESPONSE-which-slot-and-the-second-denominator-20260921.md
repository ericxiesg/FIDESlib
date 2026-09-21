# 你的第一次 he_inv 只坏了**一两个槽**——现在探针会说是哪一个

日期：2026-09-21。承接 `3775678` 的逐槽数据。**两个新探针，都是冲着"哪一个槽"去的。**

---

## 0. 把你的数据再读一遍，问题比看上去小得多

你第一次 he_inv 的三行：

```
07.inv_iter01_b  min +0.023    max +1.529   p99 0.275     <0 0/32768   >1 1/32768
07.inv_iter03_b  min -0.04582  max +61.09   p99 0.003722  <0 2/32768   >1 1/32768
```

- `min` 正的、`<0` 是 0 → **PADDING_FLOOR 生效**
- `p99` 和本机一致（0.0037）→ **32766 个槽是对的**
- `max 61.09`、`>1 1/32768` → **坏的是一个槽**

所以第一次 he_inv **不是整体发散，是一两个槽跑掉了**。
而 `07d` 的 5.7e5、第二次 he_inv 的 1e6，都是这一两个槽被后面放大的结果。

**问题从"迭代为什么发散"缩小成"哪一个槽，为什么"。**

---

## 1. 新探针 1：越界的槽在**哪里**

探针现在除了计数，还报最靠外的三个槽的下标，以及它对几何 `n_slot` 的余数——
布局是按 `n_slot` 重复的，余数通常才是有信息的那一半。

本机样例：

```
07a.refreshed_scores ... <0 86418/262144  >1 74538/262144
                         worst at 261955(%3),262003(%3),261939(%3)
                         carried max 10.21  padding max 12.43
```

三个最靠外的槽**余数都是 3**——不是随机分布。

**请把你 `inv_iter03_b` 那行的 `worst at ...` 贴回来。** 那个槽是 padding 行、
组边界、还是 slot 0，直接决定下一步查哪里。

---

## 2. 新探针 2：第二次 he_inv 的**输入**（之前没人量过）

`07c.denominator` 只覆盖**第一次** he_inv。第二次的分母在 `update_inv_D` 里，
一直没有探针——所以"第二次 iter01_b 是 1e6"到底是它自己发散、还是**接手了一个已经坏的输入**，
无法区分。

新增 `07e.halved_denominator_k<k>`。本机基线：

```
07c.denominator              max +0.3036
07e.halved_denominator_k384  max +0.3058   carried max 0.3058  padding max 0
```

**两者量级相同**——`update_inv_D` 按设计是保幅度的，`k` 就是干这个的。
本机这次 `k = 384`。

`k` 写进探针名字里，是因为下一行会把它**平方**：

```python
scaled  = multiply(_times(numerator, inverse), k)
squared = rescale(relinearize(square(scaled)))
```

`k² = 147456`。所以第一次 he_inv 输出里那一两个坏槽，到这里会被放大 **1.5e5 倍**。
这就是 `07d`（5.7e5）到第二次 `iter01_b`（7.2e6）之间那段路。

**请把 `07e` 也贴回来。** 两种结果：

| | 含义 |
|---|---|
| `07e` ≈ 0.3，只有个别槽大 | `update_inv_D` 正常，坏槽是从第一次 he_inv **继承**来的 → 查那一个槽 |
| `07e` 整体 ≈ 1e6 | `update_inv_D` 自己就放大了 → 查 `k` 和 `masked_inverse` |

`07e` 的 `carried max` / `padding max` 也会打出来（本机 padding 是精确的 0）。

---

## 3. 一起要的四个数，一次跑

```bash
THORFHE_DEBUG=1 python3 -m thorfhe.bench fhe --engine fideslib --device cuda:0 \
  --depth 37 --dnum 4 --bootstrap-level-budget 3,3 --binary-rotations \
  --refresh-after-dense --residual-scale 256 --refresh-scale 4 \
  --score-refresh-scale 2 --layers 1 --limit 1 --compact --per-stage \
  --extra-rotation-keys 6 --rotation-max-steps 4 --inverse-lift 3 \
  --plaintext-cache <cache>
```

1. `07a` 的 **carried max** 和 **padding max**（37.5 是真 token 还是 padding）
2. `07a0` 的 min/max（那次 bootstrap 是不是放大了输入）
3. `inv_iter03_b` 的 **worst at**（哪一个槽）
4. `07e` 的 max 和 carried/padding（继承还是自产）

前两个决定窗口那条线死活，后两个决定 he_inv 那条线往哪走。

221 passed, 67 skipped。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
