# "融合 stage 03/04/05 省 2 GiB" 是**过时的**——`--compact` 已经拿走了，而且峰值早就不在那里

日期：2026-09-21。修正优化清单里的一条，并指出真正的峰值在哪。

---

## 0. 背景

整模型差 2.7 GiB / 3 个 level。`he_inv` 的 epsilon 两条收紧路都被否了（`e29eb29`），
所以清单上只剩"融合 stage 03/04/05 ≈ 2 GiB"。**量之前先验证它还在不在。**

`bench workingset` 之前没有 `--compact`，所以它一直量的是**没人跑的那条路径**。加上了。
（顺带修了一个 bug：它用全零权重，方差是 3e-12，量程检查会拒——和 `plan_rotations`
一样关掉即可。这条在每层方差窗口收紧之后才暴露出来。）

---

## 1. 实测：`--compact` 已经把 QKV 那块拿走了

depth 41，一层：

| stage | 不带 compact | 带 compact |
|---|---|---|
| stage_05_value | 96 活，**3.41 GiB** | 60 活，**2.16 GiB** |
| stage_04_key | 92 活，3.28 GiB | 56 活，2.02 GiB |
| stage_03_query | 88 活，3.15 GiB | 52 活，1.89 GiB |
| **整层峰值** | **3.41 GiB**（在 stage_05） | **3.26 GiB**（在 **stage_07**） |

QKV 三个 stage 各省了约 1.25 GiB——和文档里 `stream_qkv` 的 1.07 GiB 对得上。

**但整层峰值只从 3.41 降到 3.26。** 因为峰值**已经不在 QKV 了**。

**所以融合 03/04/05 现在买不到东西。** 就算把那三个 stage 压到 0，
峰值还是 stage_07 的 3.26 GiB。这条从清单上划掉。

---

## 2. 真正的峰值：stage_07 softmax，**208 个活密文**

带 `--compact`，depth 41，按 actual 排序：

```
stage                          live      actual     if full
stage_07_softmax                208    3.26 GiB    8.53 GiB   ← 峰值
stage_06_attention_score         75    2.47 GiB    3.08 GiB
stage_08_attention_context      166    2.32 GiB    6.81 GiB
stage_05_value                   60    2.16 GiB    2.46 GiB
stage_04_key                     56    2.02 GiB    2.30 GiB
stage_03_query                   52    1.89 GiB    2.13 GiB
stage_11_attention_layernorm     61    1.67 GiB    2.50 GiB
stage_14_output_dense           114    1.49 GiB    4.68 GiB
```

`stage_07` 活着 **208** 个密文，是第二名的 2.8 倍。
`stage_08` 也有 166 个。**注意力那一段是现在的内存瓶颈，不是 QKV 投影。**

`if full` 那一列是"如果都在满 level"的大小——stage_07 是 8.53 GiB，
说明它活的那 208 个大多在**低 level**（所以 actual 只有 3.26）。
这也意味着：把它往**更低的 level** 推，收益有限；要减的是**个数**。

---

## 3. 所以最后一块内存该往哪看

| 候选 | 状态 |
|---|---|
| ~~融合 stage 03/04/05~~ | **划掉**，`--compact` 已拿走，峰值也不在那儿了 |
| **stage_07 的 208 个活密文** | **新的首要目标**，比第二名大 2.8 倍 |
| stage_08 的 166 个 | 第二 |
| `he_inv` epsilon（3 level） | 两条路都否了（`e29eb29`） |

我**没有**动 stage_07——它是 `he_softmax` 加 `update_inv_D` 的循环，
而设备正好卡在那一段的精度上。在精度定下来之前改它的结构，
会让两个问题纠缠在一起。**先等设备的四个数。**

221 passed, 67 skipped。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
