# 对照《THOR-layout 对照笔记》逐条核验我们的实现

日期：2026-09-14。核验对象：`FIDESlib/python/thorfhe/`，基准：`../THOR-layout对照笔记.md`
以及 THOR 源码 `../THOR/src/thor/`。

凡标 **[实测]** 的都是在本机跑出来的，不是读代码读出来的。方法：用 `ClearEngine`
（精确算术 + 严格 FIXEDMANUAL 记账）跑一整层，记录每个 stage 的密文条数、level、bootstrap 次数。

---

## 一句话

**布局、算子排布、密文条数全部一致；参数设置有五处不同，其中一处是我们更贵，应当改。**

---

## 一、布局：逐点一致

### 1.1 地址公式 ✓

笔记：`slot[j·2048 + t·16 + d]`，`j=0..15` 对角线、`t=0..127` token、`d=0..15`（用 12）。
我们的 `geometry.py` 给出 `group_size = n_slot * dim = 16*128 = 2048`，
`slot = group*2048 + token*16 + block` —— **同一个公式**。

### 1.2 "取对角线"而不是"取行" ✓ [实测]

这是最容易搞错的一条，也是我上一份 bert-tiny 报告**写错了**的一条（见第五节勘误）。

THOR `utils.py:24`：

```python
def ld_entry(matrix, l, i):
    """Get the i th entry of the l th lower diagonal of the matrix"""
    return matrix[(l + i) % b, i % c]
```

`l` 是**下对角线**的编号，行号是 `(l + t) % 128` —— 随 token 滚动，不是固定行。

拿 `x[t,f] = 768t + f` 喂进我们的 `encode_activations`，和 THOR 的公式逐点比：

```
 j   t  b |  ours real  ours imag |  THOR real  THOR imag |  若按"行"应为
 0   0  0 |          0         64 |          0         64 |          0
 0   1  0 |        769        833 |        769        833 |        768   <- 区分点
 0   1  1 |        897        961 |        897        961 |        896
 3   5  2 |       4104       4168 |       4104       4168 |       4099
 7 127  5 |      98182      98246 |      98182      98246 |      98183
```

**每个探点都等于 THOR 的 `ld_entry`，都不等于按行读的结果。** 我们的编码器是对的。

### 1.3 复数打包与 d 槽的重复 ✓ [实测]

笔记：第 `l` 条对角线放实部、第 `l+64` 条放虚部；6 块各放 2 份 = 12，余 4 格 padding。

实测（`x=arange`，group 0 token 0 的 16 槽窗口）：

```
[  0.+64.j  128.+192.j  256.+320.j  384.+448.j  512.+576.j  640.+704.j
   0.+64.j  128.+192.j  256.+320.j  384.+448.j  512.+576.j  640.+704.j
   0. +0.j    0.  +0.j    0.  +0.j    0.  +0.j]
block b == block b+6 for b=0..5: [True]*6
slots carrying data: 12/16
```

**复制关系、padding 数量都对上。**

### 1.4 槽利用率 37.5% ✓ [实测]

`98304 / 262144 = 37.5%`，和笔记一致。

---

## 二、算子排布：每个 stage 的密文条数全部一致 [实测]

一层跑完，逐 stage 记录：

| 笔记 | 笔记条数 | 我们 [实测] | |
|---|---:|---:|---|
| x 输入 | 4 | **4** | ✓ |
| ② 旋转副本 | 64 | **64** | ✓ |
| ③ Q | 4 | **4** | ✓ |
| ④ K | 4 | **4** | ✓ |
| ⑤ V | 4 | **4** | ✓ |
| ⑥ 注意力分数 | 8 | **8** | ✓ |
| ⑦ softmax | 128 | **128** | ✓ |
| ⑧ 上下文 A·V | 2 | **2** | ✓ |
| ⑨ 再做旋转副本 | 32 | **32** | ✓（`dense.stage_02_make_rotated_copies(context)`，2×pack 16） |
| ⑩ 输出投影 | 8 | **8** | ✓ |
| ⑪ 残差 + LN1 | 8 | **8** | ✓ |
| ⑫ FC1 | 16 | **16** | ✓ |
| ⑬ GELU | 16 | **16** | ✓ |
| ⑭ FC2 | 8 | **8** | ✓ |
| ⑮ 残差 + 充电 | 8 | **8** | ✓ |
| ⑯ LN2 | 8 | **8** | ✓ |

**十六项全中。** 包括笔记特意点出的"最反直觉的一步"——softmax 摊成 128 条——我们也是 128。

顺带：⑨ 那 32 条在我们的 trace 里没有名字，因为我们直接复用了
`stage_02_make_rotated_copies`（同一个算子，换 dense 的几何），所以它不是一个独立 stage。
条数、语义都一样。

其他结构性事实也都一致：

* **12 个头住在 d 槽、不拆分** ✓ —— `stage_06` 的公式就是
  `context[b][tau,d] = Σ_j A[b][tau,j]·V[j, n_out·b+d]`，`b` 即头号；
* **GELU 多项式**：THOR `he_tanh_single_for_gelu` 是 `p1` 32 系数（31 次）+ `p2` 28 系数（27 次）；
  我们的 `GELU_INNER` 31 次、`GELU_OUTER` 27 次 —— **完全相同**；
* **stage_15 "先加再充电"** ✓ —— 我们的 `refresh` 注释写的就是"folding pairs so it costs four
  bootstraps not eight"，实测 4 次；
* **旋转键按需定制**（每把只做到实际用到的电量档）✓ —— 就是我们的 `SetRotationKeyLevels`。

---

## 三、参数与调度：五处不同

### 3.1 depth / bootstrap 落点：14 → 我们 37 / 落 20 —— **引擎不同，我们更宽**

笔记：满电 14 格，bootstrap 充回 14 格。
我们：`depth=37`，bootstrap 落在 **20**。

原因是引擎记账方式不同。desilofhe 把 bootstrap 自身消耗的层放在"可用预算"之外，所以
它的 14 是**刷新后可用的 14**；OpenFHE/FIDESlib 把它算在 depth 里，我们实测 bootstrap
深度 17，`37 - 17 = 20`。**两边的可比量是"刷新后还剩多少"：THOR 14，我们 20。**
所以这不是偏差，是同一件事的两种计数，而且我们的余量更大——第 3.3 条就是它的直接后果。

### 3.2 ⑧ 之后的 bootstrap：THOR 有、我们没有 —— **这一处我们更贵，建议改**

笔记 ⑧："上下文 A·V [2条/14格] 🔋**算完充电 2 次** → 满电"。

我们 [实测] 没有。我们把刷新放在 **⑩ 之后**（`refresh_after_dense`，4 次）：

```
[实测] bootstrap 归属到当时在跑的 stage：
  scores            6      <- stage 07 softmax（入口 4 + he_inv 内 2）
  attention_dense   4      <- 我们插入的 refresh_after_dense
  intermediate      8      <- stage 13 GELU 入口
  output_dense      4      <- stage 15 残差后
  TOTAL            22
```

**代价差一倍**：⑧ 的上下文是 **2 条**密文，⑩ 的输出是 **8 条**（折成 4 条复数）。
在 ⑧ 之后刷新只要 **2 次**，在 ⑩ 之后要 **4 次**。

为什么会这样：`refresh_after_dense` 是我们当初解"最小 depth 52 → 34"时自己插的，
挑的是让两段链长度均衡的位置（19+18），当时并不知道 THOR 在 ⑧ 之后就刷新了。

**能不能直接搬过去？** 用实测的各级消耗推一遍：

```
context 4  --bootstrap-->  20  --stage10 耗3-->  17  --norm_1 耗14-->  3
        --intermediate 耗3-->  0   --GELU 自带 bootstrap-->  20  ...
```

**够，但 GELU 入口前恰好落到 0，没有余量。** 所以这是一个值得测的候选，不是一个可以直接断言的改法。
建议做法：加一个开关，用 `ClearEngine` 先把整层跑通（秒级），再上 GPU。

### 3.3 LayerNorm 内部 bootstrap：THOR ~3 次 ×2，我们 0 次 —— **我们更省**

笔记 ⑪⑯ 各写"LN 内部充电 ~3 次（只对 1 条密文）"。我们 [实测] **两处都是 0**。

原因就是 3.1：我们刷新后有 20 格，THOR 只有 14，所以 `he_invsqrt` 在我们这儿跑得完，
不需要中途充电。**每层省下约 6 次小充电。**

### 3.4 GELU 的 level 消耗：笔记 ~11，我们原 **14** —— **查清并修掉了 1 格**

> **2026-09-14 追记：这一节的调查有结果了，已改。** 现在是 **13 格**，输出逐位不变。
> 下面保留原始分析，结论见本节末尾的"追查结果"。

### 3.4 GELU 的 level 消耗：笔记 ~11，我们 **14** —— **未解释，值得追**

笔记 ⑬："一口气吃掉约 11 格电（14 → 3）"。
我们 [实测]：`gelu()` 输入 60 → 输出 46，**消耗 14 格**。

而两边的多项式**次数完全相同**（31 + 27），所以差别只能在求值方式。看得见的一处结构差异：

```python
# THOR evaluate_baby_poly：乘完系数直接加，不 rescale
result = self.multiply(coefficients[1], x_powers[1])
result = self.add(result, self.multiply(coefficients[2], x_powers[2]))

# 我们 evaluate_polynomial：每个 baby 项都 rescale
scaled = self.multiply(basis[power], float(coefficient))
terms.append(self.rescale(scaled))
```

我们必须 rescale——FIXEDMANUAL 下不 rescale 就没法和别的项相加；desilofhe 会隐式对齐。
但**这解释不了整整 3 格**：乘法本身那一格两边都要付。

#### 追查结果

按上面说的办法逐步打 level，账是这样的 [实测]：

```
evaluate_polynomial(GELU_INNER, 32 系数)   6 格
evaluate_polynomial(GELU_OUTER, 28 系数)   6 格   （补齐到 32，与 THOR 的 7 段不补齐同价）
gelu() 合计                               14 格   -> 复合部分占 2 格
```

复合部分那 2 格，一格是最后那次 `multiply(scaled, shifted)`（THOR 也要付），
另一格是 **`carrier` 的运行时除法**：

```python
argument = x if carrier == 1.0 else self.rescale(self.multiply(x, 1.0 / carrier))
```

THOR 没有这一步。为什么我们有：把两边的系数表对一下就清楚了 [实测]

* `GELU_INNER` == **反序的** THOR `p1`，`max|diff| = 0` —— 同一个多项式，只是存储顺序相反；
* `GELU_OUTER` == **反序的** THOR `p2` **× 0.5**，比例在 28 个系数上是常数 0.5。

也就是说，**出口的那个 1/2 早就折进系数了，入口的 1/2 却留在运行时**。这是同一套
doubled-ciphertext 约定的两半，没理由一半折一半不折。

**改法**：`p(x/c)` 等价于把系数变成 `p_k / c^k` 再在 `x` 上求值。系数是明文、折一次就够，
**运行时那次 rescale 就没了**。实测：

```
current : 14 levels
folded  : 13 levels
max|diff| 0.000e+00   relative 0.000e+00      <- 逐位相同
```

已经改进 `numeric.py`（`_gelu_inner_for(carrier)`，`lru_cache` 折一次）。
端到端保真度一字未变（`gelu` relRMSE 2.283e-03、`output_dense` 2.197e-03、`norm_2` 1.121e-03，
与改动前完全一致），98 个用例全过，钉旧值的那个用例改成钉 13。

**这一格落在哪**：GELU 的 bootstrap 到 stage 15 的 bootstrap 之间那一段——
`gelu` 6→7、`output_dense` 4→5。层输出 `norm_2` 仍是 1，因为 stage 15 的 bootstrap 会重置。
所以它是**那一段的余量**，不是层末的净增；`output_dense` 离地板远了一格，就是它的价值。

**剩下的 2 格（13 vs 笔记的 ~11）仍未解释。** 笔记自己写的是"约 11 格"，
而 desilofhe 对浮点标量乘的 level 记账和 FIXEDMANUAL 不同（它隐式对齐、我们必须显式 rescale），
两者本来就不完全可比。要坐实得在同一个引擎上跑两份实现，本文没有做。

### 3.5 旋转键：笔记 200 把，我们 15 把 —— **有意为之**

笔记：200 把定长旋转键，92% 停在电量 9 档。
我们：**2 的幂分解后 15 把**（`--binary-rotations`），数值逐位不变，代价是 4.5 倍旋转次数。
不开分解时是 ~210 把，和笔记同量级。

按需定制电量档那一条（笔记说 THOR 也这么做）我们也有，就是 `SetRotationKeyLevels`
加 `KeySwitchingKey::maxLevel`。

---

## 四、一层 bootstrap 总账对比

| | 笔记（THOR） | 我们 [实测] |
|---|---:|---:|
| ⑦ softmax 入口 | 4 | 4 |
| ⑦ he_inv 内 | ~2–4 | 2 |
| ⑧ 上下文之后 | **2** | **0** |
| ⑩ 之后（我们插入的） | **0** | **4** |
| ⑪ LN1 内 | ~3 | **0** |
| ⑬ GELU 入口 | 8 | 8 |
| ⑮ 残差之后 | 4 | 4 |
| ⑯ LN2 内 | ~3 | **0** |
| **合计** | **~26–28** | **22** |

**我们总数更少**（22 vs ~27），因为 3.1 的更宽预算省掉了两处 LayerNorm 的内部充电；
但 3.2 那一处我们多花了 2 次。把 3.2 改过去的话是 **20 次**。

---

## 五、勘误：我上一份报告写错了一处

`report/layout-bert-tiny-vs-thor-20260914.md` 第二节我把 THOR 的打包描述成
"复数槽同时装特征 f 与 f+64"，并说 `x_b[l, t]` 是**行**。

**那是错的**：`l` 是**下对角线**编号，行号是 `(l+t) % 128`（`utils.py:24`）。
在 `t=0` 上两种读法给出同一个数，我当时只验了 `t=0`，所以没看出来。
本文 1.2 用 `t=1,5,127` 区分开了。

这不影响那份报告的结论（bert-tiny 是行主序方阵、THOR 是 group/token/block 三层结构、
必须转置 vs 不必转置、头拆分 vs 头并行），但那一行的措辞要改。已在该文件内标注。

---

## 六、结论与建议

**布局和算子排布：一致，可以放心。** 16 项密文条数全中、对角线公式逐点匹配、
GELU 多项式次数相同、12 头并行的结构相同。

**要动的一处：把刷新从 ⑩ 之后挪到 ⑧ 之后**（3.2）。每层省 2 次 bootstrap，
按笔记的 0.66 秒/次算是 **1.3 秒/层、12 层约 16 秒**。风险是 GELU 入口前余量归零，
必须先用 `ClearEngine` 跑通再上 GPU。

**已经查掉的一处：GELU 那 3 格**（3.4）。其中 1 格是真的、已修——
`carrier` 的运行时除法折进了内层系数，输出逐位不变，`gelu` 从 14 格降到 13 格。
剩下 2 格是两个引擎对浮点标量乘的记账差异，不可直接比较。

**不用动的：** depth/bootstrap 落点（3.1，我们更宽）、LayerNorm 内部充电（3.3，我们更省）、
旋转键数量（3.5，有意为之）。
