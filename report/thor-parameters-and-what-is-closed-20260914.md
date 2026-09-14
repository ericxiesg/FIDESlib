# THOR 用的是什么参数、哪些是闭源的，以及我们的 L 为什么必须 ≥ 36

日期：2026-09-14。源码：`../THOR/`（开源部分）、`desilofhe-cu130` wheel（闭源部分）、
本仓库 `python/thorfhe/`。凡标 **[实测]** 的是在本机跑出来的。

---

## 一、最重要的一条：**THOR 根本不选 CKKS 参数**

`he.py:36` 建引擎的全部代码是：

```python
self.engine = Engine(use_bootstrap_to_14_levels=True, mode="async gpu",
                     device_id=device, compact=compact)
```

`encode_weights.py:354` 同理。**没有 `N`、没有 `L`、没有缩放位宽、没有 `dnum`、
没有特殊素数个数、没有 bootstrap 的 level budget。** 唯一和 level 有关的旋钮是
`use_bootstrap_to_14_levels=True`。

而 `max_level` 是**读出来的**，不是设进去的：

```python
max_level = self.engine.max_level          # he.py:80
```

**结论：THOR 没有"L 是多少"这个概念可供对照。** L 由 desilofhe 内部按
`use_bootstrap_to_14_levels` 推出来，THOR 只知道"刷新后有 14 格可用"。
所以拿我们的 `L=23 / 25 / 37` 去和 THOR 比，是没有对手的——**那是我们自己的工程选择**。

---

## 二、开源 / 闭源的分界

THOR 仓库一共 3193 行 Python，依赖里只有一个非常规项：

```toml
dependencies = [ ..., "desilofhe-cu130>=1.14.0,<2.0.0" ]
```

| | 谁做的 | 具体内容 |
|---|---|---|
| **开源**（THOR 仓库，可读可改） | THOR | 打包布局与 `encrypt_embedding`；18 个 stage 的调度；多项式系数表（`p1/p2`、`he_exp1/2`、pooler）；掩码表（`rotate_internal`、`transpose`、`ccmm`、`make_copies`）；**`rotation_contexts` 那张 200 项的 (delta, level) 表**；compact/large 两套 `bootstrap_deltas` |
| **闭源**（desilofhe wheel，只有二进制） | Desilo | **整个 CKKS 引擎**：参数选择（N、L、Δ、dnum、特殊素数）、bootstrap 实现及其 level budget、key switching、rescale 策略、NTT/RNS 全部内核，以及 `max_level` 本身 |

换句话说：**THOR 开源的是"怎么摆数据、按什么顺序算"，闭源的是"算得对不对、多快、要几格电"。**

这对我们有两个直接后果：

1. **我们移植的是可见的那一半**，另一半我们用 FIDESlib/OpenFHE 重新做——
   所以两边的 level 记账本来就不可能逐格对上（见 `thor-note-conformance-20260914.md` 第 3.1、3.4 节）；
2. **那张 `rotation_contexts` 表是白送的**：200 项 (delta, level)，是 THOR 跑一遍量出来的，
   正好就是我们 `SetRotationKeyLevels` 要的东西。它是开源部分里最实用的一块。

### 顺带：THOR 那张表长什么样

```python
rotation_contexts = [(0, 9), (1, 14), (2, 14), (3, 12), (4, 14), (5, 12), (6, 11), (8, 7),
                     (16, 10), (32, 10), ..., (32766, 13), (32767, 13)]
```

`(delta, level)` = 这个平移量实际会在多高的 level 上被用到。落在 `bootstrap_deltas`
里的不单独配钥匙（打包进 bootstrap key）。**92% 停在 level 9** 的说法就是从这张表数出来的。

---

## 三、我们的 L 为什么必须 ≥ 36 [实测]

这是我们自己的账，和 THOR 无关。用 `ClearEngine` 跑一整层（`depth=37`、bootstrap 落 20、
开 `refresh_after_dense`），记录每个 stage 的 level，**两次 bootstrap 之间的那一段**就是约束：

| 段 | 起点 | 段内最深处 | 段耗 |
|---|---:|---:|---:|
| 入口 → softmax 的 bootstrap | 37 | `scores` 25 | 12 |
| softmax 的 bootstrap → refresh | 20 | `attention_dense` **1** | **19** ← 约束 |
| refresh → GELU 的 bootstrap | 20 | `intermediate` 3 | 17 |
| GELU 的 bootstrap → stage 15 | 20 | `output_dense` 5 | 15 |
| stage 15 的 bootstrap → 层末 | 17 | `norm_2` 1 | 16 |

**最深的一段要 19 格**（softmax 之后一路到 attention dense）。而

```
bootstrap 落点 = L - bootstrap_depth = L - 17      # bootstrap_depth 17 是实测值，budget.py
```

所以 `L - 17 ≥ 19` → **L ≥ 36**。我们跑 37，留一格余量。

（GELU 那一格优化之后，`gelu` 6→7、`output_dense` 4→5，第四段从 15 降到 14；
但约束在第二段，所以最小 L 没变。）

### 那么 L=23 / L=25 会怎样

按同一个公式：

| L | bootstrap 落点 | 第二段需要 19 格 | 缺 |
|---:|---:|---:|---:|
| 23 | 6 | 19 | **13** |
| 25 | 8 | 19 | **11** |
| 36 | 19 | 19 | 0（零余量） |
| 37 | 20 | 19 | 1 |

**两个都远远不够**，整层不可能跑完。差别只在"从哪一步开始耗尽"——
L=25 比 L=23 多撑两格，所以耗尽的位置不同，**症状也就不同**。

---

## 四、L=23 crash / L=25 解码失败：我还不能回答

上面的算术说明"两个都该失败"，但**说明不了为什么一个是 crash、另一个是解码错**。
这个仓库里没有这两次运行的任何记录（`bugs/`、`report/` 都搜过，远程也没有新报告）。

而这正是本项目反复吃亏的地方——同一个 softmax 的 bug 我连猜错三次方向，
每次都是实测把我纠回来。所以这次不猜。

**要回答需要三样东西**（有哪个给哪个）：

1. **完整报错文本**——尤其是不是我们自己加的那几条诊断之一：
   * `EvalLevelReduce would drop every RNS limb`（level 用尽，我们抛的）
   * `FIDESlib: addPt/subPt with mismatched scales ...`（scale 不匹配，我们抛的）
   * `LTdotProductPtBatch would read N limbs from pt[k], which holds M`（limb 不匹配，我们抛的）
   * `this parameter set needs ... MAXP = 64`（常量表溢出，我们抛的）
   * 以上都不是的话，大概率是 CUDA illegal access 或 OpenFHE 自己抛的；
2. **跑的是哪条命令**——整层 benchmark，还是 `tests/` 里某个用例，还是裸 bootstrap；
3. **配套的 `--device-memory` 与逐级表**（如果是 benchmark）。

有了 1 和 2 基本就能定位：如果报的是 `EvalLevelReduce would drop every RNS limb`，
那就是上表那个缺 11/13 格，和第三节完全一致，没有别的故事；
如果报的是别的，那就是**另一个独立问题**，得单独查。
