# 排版对比：FIDESlib `examples/bert-tiny` 与 THOR 的单层 transformer

日期：2026-09-14。

对比三方：

| | 位置 | 语言 |
|---|---|---|
| **bert-tiny** | `FIDESlib/examples/bert-tiny/src/` | CUDA / C++，4263 行 |
| **THOR 原版** | `../THOR/src/thor/` | Python + desilofhe，3193 行 |
| **本项目移植** | `FIDESlib/python/thorfhe/` | Python + pyfideslib |

本文结论里凡标 **[实测]** 的都是在本机跑出来的数字，其余是读代码得到的。

---

## 一、一句话

**两者的矩阵乘法属于两个不同的算法族，差别不在实现细节而在打包。**
bert-tiny 用的是 JKLS（σ/τ/φ 置换 + 方阵）那一族，一个密文装一个 `n×n` 方阵、行主序；
THOR 用的是**块对角 + 窗口内旋转**那一族，一个密文装
`group × token × block` 三层结构，靠预编码成对角线的权重和窗口内循环移位来对齐。

直接后果：**bert-tiny 必须显式做矩阵转置，THOR 不需要**；
**bert-tiny 必须把多头拆成多组密文分别算，THOR 十二个头同时算**。

---

## 二、打包：这是所有差别的根

### bert-tiny

```
blockSize = sqrt(numSlots)          // main/berttiny-all.cu:33
slot = row * blockSize + col
```

一个密文 = 一个 `blockSize × blockSize` 方阵，**行主序、槽全满**。
大于 `blockSize` 的矩阵切成瓦片，类型就是 `std::vector<std::vector<Ciphertext>>`（瓦片网格）。
bert-tiny 隐藏维 128、`numSlots = 2^14` 时 `blockSize = 128`，所以
128 token × 128 特征的激活**正好一个密文**。

### THOR

```
slot = group * 2048 + token * 16 + block           # group_size=2048, n_slot=16
复数槽同时装特征 f 与 f + 64                        # n_in/2 = 64
```

`data_encoder.py:41` 的 `encrypt_embedding` 把 `(128, 768)` 转置后 `vsplit` 成 6 片，
再按上式填入 **4 个密文**。

**[实测]** 用我们的 `encode_activations` 解回来（`python -m` 一次性脚本）：

```
ciphertexts: 4    slots each: 32768 (complex)
token window, group 0 token 0:
  [  0.+64.j  128.+192.j  256.+320.j  384.+448.j  512.+576.j  640.+704.j
     0.+64.j  128.+192.j  256.+320.j  384.+448.j  512.+576.j  640.+704.j
     0. +0.j    0.  +0.j    0.  +0.j    0.  +0.j]
block b equals block b+6 for b=0..5: [True, True, True, True, True, True]
slots carrying data: 12/16; real values 98304 of capacity 262144 -> density 37.5%
```

三件事同时成立，都是有意的：

1. **复数槽装两个特征**（`f` 与 `f+64`）——把密文数减半；
2. **block `b` 与 `b+6` 完全重复**——同一份数据在 16 槽窗口里存两遍；
3. **16 槽只用 12 槽**，4 槽是留白。

### 密度对比

| | 一个 128×768 激活需要 | 槽利用率 | 折算成 `2^14` 槽的单位 |
|---|---|---|---|
| bert-tiny | 6 个密文（`N=2^15`，16384 实槽） | **100%** | 6 |
| THOR | 4 个密文（`N=2^16`，32768 复槽） | **37.5%** [实测] | 8 |

**THOR 为同样的激活付出约 1.33 倍的密文体积。** 换来的是第三节那三件事。
这不是浪费——重复和留白正是窗口内旋转能在不污染相邻 token 的前提下工作的原因——
但它是真实成本，讨论内存时要算进去。

---

## 三、矩阵重排：四处关键差别

### 3.1 明文-密文乘（PCMM）：置换两个操作数 vs 只转权重

**bert-tiny**（`MatMul.cuh`）：`sigmaPlaintexts`、`tauPlaintexts`、`phiPlaintexts` —— 这是
JKLS 的 σ/τ/φ/ψ 置换。矩阵乘之前**两个操作数都要先被线性变换置换一遍**，
每个置换是一次对角线线性变换（BSGS，`bStep=16`）。

**THOR**（`he.py:253 parallel_diagonal_pc_mult` / `:292 pcmm`）：

```python
temp = multiply(ws[out, diag, 0], prepared[(16*out) % in_dim])
for in_index in 1..in_dim-1:
    temp += multiply(ws[out, diag, in_index], prepared[(16*out + in_index) % in_dim])
out = rescale(temp)
# 然后把 diag_dim 个部分结果用 rotate_internal 对齐相加
```

权重**离线就编码成块对角线**（`encode_to_light_plaintext`），密文一次都不置换——
被"旋转"的是**密文数组的下标** `(16*out + in) % in_dim`，那是免费的 Python 索引。
真正的密文重排只发生在 `rotate_internal`：

```python
masked  = multiply(mask, x)
right   = rotate(masked, right_delta)         # 窗口内回绕的那部分
left    = rotate(x - masked, -left_delta)     # 其余部分
return left + right
```

**一次掩码 + 两次旋转 + 一次加**，窗口宽度 12（`block_diag_1`）或 6（`block_diag_2`）。
这就是为什么 `n_slot=16 > n_blocks=12`：留白吸收回绕。

> **差别的实质**：bert-tiny 把"对齐"做成**对数据的置换**；THOR 把它做成
> **对权重的预排列 + 窗口内的小旋转**。前者通用（任意方阵），后者只对它自己的打包成立，
> 但省掉了对密文的全局置换。

### 3.2 转置：显式线性变换 vs 打包置换

**bert-tiny** 有一整个 `Transpose.cu`：`MatrixTransposeSquare_GPU` 用
`diagPlaintexts`（`2*rowSize-1` 条对角线）+ BSGS 做一次完整的同态矩阵转置，
自带一套旋转键（`GenerateTransposeRotationIndices_GPU`）。算 `QKᵀ` 前**必须**对 K 做：

```cpp
auto K1_T = MatrixTranspose_GPU(std::move(K1), conf.blockSize, Tprecomp_gpu);
auto K2_T = MatrixTranspose_GPU(std::move(K2), conf.blockSize, Tprecomp_gpu);
CCMM_GPU(Q1, K1_T, conf.blockSize, QKT1, precomp_gpu);
```

**THOR** 的 `transpose_upper_to_lower`（`he.py:362`）不是矩阵转置，是**打包的重排**：
在它的布局里，"转置"退化成一个有结构的槽置换，用 4 条掩码
（`m0..m3`，`transpose_masks`）加若干旋转就能完成，**代价约一个 level**。

我们移植版的公式写在 `attention.py:112`：

```
out[ct][group, tau, b] = k[t, f]
ct * pack + group = (t - d) mod n_out
tau = d + n_out * ((d > t mod n_out) XOR (t // n_out))
其中 d = f % n_out, b = f // n_out
```

> **差别的实质**：bert-tiny 的转置是 O(n) 条对角线的通用线性变换；
> THOR 的"转置"是把它布局里本来就存在的对称性用掩码摘出来。
> **代价差一个数量级，通用性也差一个数量级。**

### 3.3 密文-密文乘（CC-MM）：φ/ψ 旋转 vs `make_copies` 广播

**bert-tiny**：JKLS 的 `sum_k φ^k(σ(A)) ⊙ ψ^k(τ(B))`，两个操作数都在旋转。

**THOR**（`he.py:406 make_copies` + `:447 stage_06_attention_score`）：
把**一个**操作数的每条对角线**广播到所有 group**，另一个操作数只做整体旋转：

```python
copies[l][group, t, b] = x[t, n_out*b + (l + t) mod n_out] / 2     # 对所有 group 相同
```

`make_copies` 本身花 2 个 level（chunk 掩码 + 半组掩码），产出 `n_out = 64` 条对角线；
然后 64 次乘加，每次把 `left` 旋转 `group_size*j - n_slot*in_index`，
再用 `ccmm` 的 4 条掩码把乘积切成四份累加。

> **差别的实质**：bert-tiny 对称地旋转两个操作数；THOR 不对称——
> 一个操作数被**展开成 64 条广播对角线**（这就是我们之前那 1.8 GiB 工作集的来源），
> 另一个只做整体旋转。

### 3.4 多头：显式拆分 vs block 维并行

**bert-tiny**（`Transformer.cu`，`num_heads = 2`）：

```cpp
// Q1, Q2, K1, K2 是分开的密文矩阵
CCMM_GPU(Q1, K1_T, ..., QKT1, ...);
CCMM_GPU(Q2, K2_T, ..., QKT2, ...);
EvalSoftmax_Matrix(QKT1, ...);
EvalSoftmax_Matrix(QKT2, ...);
```

**每个头一套完整流程**，BERT-base 的 12 个头就是 12 遍。

**THOR**：头的下标就是 **block 维**。`n_out = 64` 是头内维度，`n_blocks = 12` 是 12 个头，
`context[b][tau, d] = sum_j A[b][tau, j] * V[j, n_out*b + d]` —— **b 就是头**。
12 个头在同一批密文里同时算，softmax 也只做一次。

> 这是 THOR 那个"低密度"打包换来的最大一笔：**头并行是免费的**。

---

## 四、完整算子对照（单层）

| 步骤 | bert-tiny | THOR / 本移植 |
|---|---|---|
| 输入 | 瓦片网格，行主序方阵 | 4 密文，`group/token/block`，复数打包 |
| — | — | `stage_01_complexify_x`（拆实虚部 / 折叠副本） |
| — | — | `stage_02_make_rotated_copies`（块对角旋转副本） |
| Q/K/V | `PCMM_GPU` ×3（σ/τ 置换 + BSGS，带 bias 与 row mask） | `stage_03/04/05`（块对角 `pcmm`，权重离线成对角线） |
| 刷新 | `MatrixBootstrap(Q)`、`(K)`、`(V)` —— **3 次，在这里** | 无（THOR 把 bootstrap 放进 softmax） |
| 分头 | 显式切成 `Q1,Q2 / K1,K2` | 无（头在 block 维） |
| 转置 | `MatrixTranspose_GPU(K)` **每头一次** | `transpose_upper_to_lower`（打包置换，一次） |
| QKᵀ | `CCMM_GPU` **每头一次** | `stage_06_attention_score`（`make_copies` + 64 次乘加） |
| 刷新 | `MatrixBootstrap(QKT, prescale=true)` | `stage_07` 内部的 bootstrap |
| softmax | `EvalSoftmax_Matrix` **每头一次** | `stage_07_softmax`（一次，覆盖全部头） |
| A·V | `CCMM_GPU` 每头一次 | `stage_08_attention_context`（`consume` 流式） |
| 输出投影 | `PCMM_GPU` | `stage_10_attention_dense`（换成 6-block 表示） |
| — | — | `refresh`（**本项目新增**，把 37 级链切成 19+18） |
| LayerNorm | `EvalLayerNorm_Matrix` | `stage_11_attention_layernorm` |
| FFN 上投影 | `PCMM_GPU` | `stage_12_intermediate_dense` |
| GELU | `EvalGelu_Matrix` | `stage_13_gelu`（两段多项式复合） |
| FFN 下投影 | `PCMM_GPU` | `stage_14_output_dense` |
| LayerNorm | `EvalLayerNorm_Matrix` | `stage_15/16` |

---

## 五、非线性算子：思路其实很接近

这一块两边比矩阵那块像得多。

| | bert-tiny | THOR |
|---|---|---|
| exp | Chebyshev 系数 `cheb_coeff_exp_softmax_1`，预缩放到 `[-1,1]` | 15 次 minimax 拟合 `EXP1/EXP2_COEFFICIENTS` |
| 锐化 | 反复平方 `x² → … → x^16`（`PolyApprox.cu:485-491`） | `n` 次平方 + `update_inv_D` 每次再折半温度 |
| 1/x | Chebyshev 拟合 `cheb_coeff_inv_softmax1_59` | **Goldschmidt 迭代**（`he_inv`，带 `DeltaCiphertext` 整数缩放） |
| 行求和 | `AccumulateBroadcast` / `rotsum_GPU` | `interval_sum` |
| 1/sqrt | `NewtonRaphsonInvSqrt` | `he_invsqrt`（Goldschmidt） |
| 多项式求值 | `evalFunction`（Chebyshev） | `evaluate_polynomial`（BSGS / Stockmeyer，baby degree 3） |

**共同的核心手法一致**：算低温 softmax 再靠平方锐化。
差别在倒数——bert-tiny 直接拟合 `1/x`，THOR 用迭代法并把尺度作为**整数**拉回
（整数乘不耗 level，这正是 `DeltaCiphertext` 的用途）。

---

## 六、level 与 bootstrap 的放法

* **bert-tiny**：Q/K/V 投影后立刻 3 次 bootstrap，QKᵀ 后再 1 次，
  即**每层至少 4 次矩阵级 bootstrap**（矩阵级 = 瓦片数 × 次数）。
* **THOR**：整层只有 softmax 内部那一次；本项目为了让 37 级预算塞下，
  额外插了一次 `refresh_after_dense`（值语义中性，把最深的链切成 19+18）。

---

## 七、旋转键

* **bert-tiny**：`GenerateMatMulRotationIndices_GPU` + `GenerateTransposeRotationIndices_GPU`，
  按 BSGS 的 `bStep=16` 生成两套，**转置那套是额外的**。
* **THOR**：约 250 把固定 level 的旋转键；本项目改成 **2 的幂分解后只要 15 把**
  （51 GiB → 3.6 GiB，代价 4.5 倍旋转次数）。

---

## 八、可借鉴的地方

按"值得先看"排：

1. **bert-tiny 的瓦片抽象**（`vector<vector<Ciphertext>>` + `MatrixBootstrap` / `MatrixAddScalar`
   这类矩阵级算子）比 THOR 的裸 `np.empty((n,), dtype=object)` 好读得多，
   而且天然支持任意大小的矩阵。THOR 的打包是**为 BERT-base 的形状手工裁的**
   （`require_thor_shape` 就是在说这件事）。
2. **bert-tiny 的 softmax 用统计量归一化**（`mask_max`、`num_sigma`），
   THOR 用**离线标定的固定窗口**（`min_x/max_x` 表）。我们已经吃过固定窗口的亏——
   窗口外不是精度下降而是发散。前者更稳，代价是多几次规约。
3. **转置**：如果以后要支持 THOR 打包之外的形状，bert-tiny 那套通用转置是现成的，
   而且它已经在 FIDESlib 里、用的是同一套 `LinearTransform` 设施。

反过来，**THOR 值得守住的**是头并行和块对角 PCMM：
12 个头一次算完、权重零运行时置换，这两点 bert-tiny 没有，而且是 THOR 吞吐的主要来源。

---

## 九、没有核实的部分

* bert-tiny 的 `EvalSoftmax` 我只读到"Chebyshev exp + 反复平方 + Chebyshev 倒数"这一层，
  `mask_max` / `num_sigma` 具体怎么参与归一化没有逐行确认。
* 两边的**实测吞吐没有对比过**——bert-tiny 是 BERT-tiny（2 层、隐藏 128、2 头），
  THOR 是 BERT-base（12 层、隐藏 768、12 头），直接比时间没有意义，
  要比得把 bert-tiny 的配置拉到 BERT-base 或反之，本文没有做。
* 第二节那张密度表里 bert-tiny 一侧的"6 个密文"是按 `blockSize=128` 推的，
  不是跑出来的；THOR 一侧是 **[实测]**。
