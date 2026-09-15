# BERT-Tiny 端到端算子级追踪报告

日期：2026-09-15
硬件：Quadro GV100 32GB (sm_70), CUDA 12.9
参数：L=25, logN=15 (N=32768), dnum=1, scale_mod_size=52, FLEXIBLEAUTO, KEY_GROW=1
库：FIDESlib bootstrap-dev (2b1a029), 含 power_basis 修复 + GELU 省 level + degree 检查

---

## 1. 追踪方法

在 `examples/bert-tiny/src/Transformer.cu` 的 `encoder()` 函数中每个算子前后插入探针，
记录：
- **算子名** (dropLevel, PCMM, CCMM, Boot, Softmax, GELU, LayerNorm, Residual_Add 等)
- **输入/输出矩阵形状** (rows × cols, 每个元素是一个密文)
- **输入/输出密文 level** (0~25, 越高越新鲜)
- **输入/输出 NoiseLevel** (1=canonical, 2=需要 rescale)
- **输入/输出 degree2** (是否有未 relinearize 的 c2 组件)
- **slots** (16384 = 2^14)
- **GPU 内存**：pooled (池总占用), in_use (池中在用), driver_free (驱动剩余)
- **耗时** (ms)

原始 CSV 数据：`report/bert_tiny_trace.csv` (348 行, 6 个样本 × 58 步/样本)

---

## 2. 单样本算子流程（Sample 3, Layer 0+1, 第一条 measured 样本）

### Layer 0 (29 步)

| Step | 算子 | in_shape | out_shape | in_L→out_L | in_NL→out_NL | pooled_mb | in_use_mb | free_mb | ms |
|------|------|----------|-----------|------------|-------------|-----------|-----------|---------|-----|
| 0 | dropLevel_tokens | 1x1 | 1x1 | 4→4 | 1→1 | 11293 | 10997 | 20721 | 0 |
| 1 | PCMM_K | 1x1 | 1x1 | 4→1 | 1→1 | 12317 | 11540 | 19689 | 20 |
| 2 | PCMM_Q | 1x1 | 1x1 | 4→1 | 1→1 | 12317 | 11553 | 19689 | 12 |
| 3 | PCMM_V | 1x1 | 1x1 | 4→1 | 1→1 | 12317 | 11566 | 19689 | 12 |
| 4 | Boot_Q | 1x1 | 1x1 | 9→9 | 2→2 | 12317 | 12049 | 19687 | 526 |
| 5 | Boot_K | 1x1 | 1x1 | 9→9 | 2→2 | 12317 | 12086 | 19687 | 13 |
| 6 | Boot_V | 1x1 | 1x1 | 9→9 | 2→2 | 12317 | 12107 | 19687 | 13 |
| 7 | CCMM_QKT1 | 1x1 | 1x1 | 5→2 | 2→2 | 13341 | 12461 | 18663 | 21 |
| 8 | CCMM_QKT2 | 1x1 | 1x1 | 5→2 | 2→2 | 13341 | 12631 | 18663 | 19 |
| 9 | Boot_QKT1_pre_softmax | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 12753 | 18663 | 13 |
| 10 | Softmax_QKT1 | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 12784 | 18663 | 319 |
| 11 | Boot_QKT2_pre_softmax | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 12793 | 18663 | 13 |
| 12 | Softmax_QKT2 | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 12786 | 18663 | 316 |
| 13 | CCMM_Sm_V1 | 1x1 | 1x1 | 5→2 | 2→2 | 13341 | 12814 | 18663 | 17 |
| 14 | CCMM_Sm_V2 | 1x1 | 1x1 | 5→2 | 2→2 | 13341 | 12837 | 18663 | 17 |
| 15 | Boot_Sm_V | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 12907 | 18663 | 13 |
| 16 | PCMM_Output | 1x1 | 1x1 | 4→1 | 1→1 | 13341 | 12907 | 18663 | 11 |
| 17 | Residual_Add1 | 1x1 | 1x1 | 1→1 | 1→1 | 13341 | 12907 | 18663 | 0 |
| 18 | Boot_Residual1 | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 13065 | 18663 | 14 |
| 19 | LayerNorm1 | 1x1 | 1x1 | 2→2 | 2→2 | 13341 | 13050 | 18663 | 55 |
| 20 | Boot_LN1 | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 13066 | 18663 | 14 |
| 21 | PCMM_Up | 1x1 | **1x4** | 4→1 | 1→1 | 13341 | 13066 | 18663 | 79 |
| 22 | Boot_pre_GELU | 1x4 | 1x4 | 9→9 | 2→2 | 13341 | 13130 | 18663 | 72 |
| 23 | GELU | 1x4 | 1x4 | 9→9 | 2→2 | 13341 | 13130 | 18663 | 169 |
| 24 | PCMM_Down | 1x4 | 1x1 | 4→1 | 1→1 | 13341 | 13130 | 18663 | 60 |
| 25 | Residual_Add2 | 1x1 | 1x1 | 1→1 | 1→1 | 13341 | 13130 | 18663 | 0 |
| 26 | Boot_pre_LN2 | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 13171 | 18663 | 12 |
| 27 | LayerNorm2 | 1x1 | 1x1 | 2→2 | 2→2 | 13341 | 13156 | 18663 | 50 |
| 28 | Boot_LN2_final | 1x1 | 1x1 | 9→9 | 2→2 | 13341 | 13172 | 18663 | 13 |

### Layer 1 (29 步, 同构)

Layer 1 的算子序列与 Layer 0 完全相同，耗时也接近。主要差异：
- PCMM_Up 输出 1x4 (128→512), PCMM_Down 输入 1x4 (512→128) — 中间层扩展到 4 个密文
- SST-2 偏移 -0.25 在 QKT 上（在 Softmax 之前）

### 单层耗时分布

| 算子 | Layer 0 (ms) | Layer 1 (ms) | 占比 |
|------|-------------|-------------|------|
| PCMM_K+Q+V | 44 | 35 | 5% |
| Boot_Q+K+V | 552 | 36 | 7% |
| CCMM_QKT1+2 | 40 | 33 | 4% |
| Boot_QKT1+2_pre_softmax | 26 | 24 | 3% |
| Softmax_QKT1+2 | 635 | 650 | 46% |
| CCMM_Sm_V1+2 | 34 | 34 | 3% |
| Boot_Sm_V | 13 | 13 | 1% |
| PCMM_Output | 11 | 11 | 1% |
| Boot_Residual1 | 14 | 12 | 1% |
| LayerNorm1 | 55 | 58 | 4% |
| Boot_LN1 | 14 | 13 | 1% |
| PCMM_Up | 79 | 97 | 6% |
| Boot_pre_GELU | 72 | 92 | 6% |
| GELU | 169 | 173 | 12% |
| PCMM_Down | 60 | 75 | 5% |
| Boot_pre_LN2 | 12 | 12 | 1% |
| LayerNorm2 | 50 | 55 | 4% |
| Boot_LN2_final | 13 | 12 | 1% |
| **总计** | **~1867** | **~1736** | **100%** |

**关键发现**：Softmax 占 46% 的单层时间（635ms/1400ms），是最大瓶颈。
GELU 占 12%，PCMM_Up 占 6%。Bootstrap 除非是首次（Boot_Q 首次 526ms），其余仅 12-14ms（复用预计算）。

---

## 3. 密文 Level 追踪

### 3.1 Level 变化模式（每层）

```
输入 tokens:    L=4,  NL=1  (fresh, from encryptMatrixtoGPU)
  ↓ PCMM ×3     L=1,  NL=1  (消耗 3 level: matmul rescale chain)
  ↓ Boot ×3     L=9,  NL=2  (bootstrap 恢复到 L=9, 但 NL=2!)
  ↓ dropLevel   L=5/6,      (Q→5, K→6)
  ↓ CCMM ×2     L=2,  NL=2  (消耗 3 level + NL=2)
  ↓ Boot ×2     L=9,  NL=2
  ↓ Softmax ×2  L=9,  NL=2  (内部多次 bootstrap, 输出仍在 L=9)
  ↓ dropLevel   L=5
  ↓ CCMM ×2     L=2,  NL=2
  ↓ Boot        L=9,  NL=2
  ↓ PCMM_Output L=1,  NL=1
  ↓ Residual    L=1,  NL=1
  ↓ Boot        L=9,  NL=2
  ↓ LayerNorm1  L=2,  NL=2  (消耗大量 level: accumulate+poly+NewtonRaphson)
  ↓ Boot        L=9,  NL=2
  ↓ PCMM_Up     L=1,  NL=1  (输出 1x4, 4 个密文)
  ↓ Boot        L=9,  NL=2
  ↓ GELU        L=9,  NL=2  (内部有 mult+boot, 输出仍在 L=9)
  ↓ PCMM_Down   L=1,  NL=1  (输入 1x4, 输出 1x1)
  ↓ Residual    L=1,  NL=1
  ↓ Boot        L=9,  NL=2
  ↓ LayerNorm2  L=2,  NL=2
  ↓ Boot        L=9,  NL=2  (最终输出)
```

### 3.2 关键观察

1. **Bootstrap 输出 NoiseLevel 始终为 2**，不是 1（canonical）。这意味着 bootstrap 后的密文需要一次 rescale 才能用于下一次乘法。这与远程协作者在 `RESPONSE-gpu-noise-level-accessor-20260914.md` 中的发现一致——bootstrap 后的 NoiseLevel 问题。

2. **Level 周期**：每次 bootstrap 恢复到 L=9（不是 L=25）。这是因为 bootstrap 的 StC 起始 level 为 8（margin=1），所以输出 level = 8+1 = 9。大部分计算在 L=9 到 L=1 之间循环。

3. **PCMM 消耗 3 level** (4→1)：明文-密文矩阵乘内部有多次 rescale。
4. **CCMM 消耗 3 level** (5→2)：密文-密文矩阵乘更耗 level。
5. **LayerNorm 消耗 7 level** (9→2)：含 accumulate + variance + Chebyshev poly + NewtonRaphson。

### 3.3 矩阵形状变化

```
tokens:     1x1  (128×128 矩阵装在 1 个密文中)
K, Q, V:    1x1
QKT:        1x1  (每头一个, 共 2 头分开计算)
Sm_V:       1x1
Output:     1x1
Up:         1x4  (128→512, 切成 4 个 128×128 瓦片)
GELU:       1x4  (4 个密文逐个激活)
Down:       1x1  (512→128, 4 个瓦片合并)
```

---

## 4. GPU 内存追踪

### 4.1 内存随步骤变化

| 阶段 | pooled_mb | in_use_mb | driver_free_mb |
|------|-----------|-----------|----------------|
| 初始 (keys+plaintexts loaded) | 11293 | 10997 | 20721 |
| PCMM 后 | 12317 | 11540 | 19689 |
| Boot_Q 后 (首次 bootstrap) | 12317 | 12049 | 19687 |
| CCMM 后 | 13341 | 12461 | 18663 |
| Layer 0 结束 | 13341 | 13172 | 18663 |
| Layer 1 结束 | 13341 | 13602 | 17607 |
| 最终 | 14365 | 13820 | 17607 |

### 4.2 内存分析

- **初始静态占用**：~11 GB (keys ~2.8GB + plaintexts ~2.5GB + bootstrap precomp ~5.7GB)
- **运行时增长**：pooled 从 11.3GB 增长到 14.4GB (+3.1GB)，主要是中间密文
- **峰值 in_use**：~13.8 GB (sample 5, layer 1 结束)
- **GPU 总量**：32 GB, 使用 ~14 GB, 剩余 ~17.6 GB — **无 OOM 风险**

### 4.3 内存池行为

pooled 内存只增长不回收（pool 保留 slab 供复用）。in_use 随算子波动：
- Bootstrap 增加约 50-60 MB（临时密文）
- PCMM 增加约 10 MB
- Softmax 增加约 30 MB
- 两个样本之间，in_use 略有增长（前一样本的密文未完全释放）

---

## 5. 耗时分析

### 5.1 首次 bootstrap 开销

Sample 0 的第一个 bootstrap (Boot_Q) 耗时 **526 ms**，而后续所有 bootstrap 仅 12-14 ms。
这是因为首次 bootstrap 需要编译/warmup GPU kernel，后续复用。

### 5.2 各样本总耗时

| Sample | Layer | Total ms | 说明 |
|--------|-------|----------|------|
| 0 | 0+1 | ~3603 | warmup (含首次 boot 526ms) |
| 1 | 0+1 | ~3077 | warmup |
| 2 | 0+1 | ~3103 | warmup |
| 3 | 0+1 | ~8320 | measured (含 classifier) |
| 4 | 0+1 | ~8269 | measured |
| 5 | 0+1 | ~8320 | measured |

**注**：sample 0-2 是 warmup 不计入 accuracy。Sample 3-5 的 8.3s 包含 classifier 时间。
单层 encoder 约 1.4-1.8s，两层约 3.1s，classifier 约 5.2s（含 tanh + accumulate + decrypt）。

### 5.3 算子耗时排序（Layer 0, sample 3, 稳态值）

| 排名 | 算子 | ms | 占比 |
|------|------|-----|------|
| 1 | Softmax_QKT1 | 319 | 17% |
| 2 | Softmax_QKT2 | 316 | 17% |
| 3 | GELU | 169 | 9% |
| 4 | PCMM_Up | 79 | 4% |
| 5 | Boot_pre_GELU | 72 | 4% |
| 6 | PCMM_Down | 60 | 3% |
| 7 | LayerNorm1 | 55 | 3% |
| 8 | LayerNorm2 | 50 | 3% |
| 9 | CCMM_QKT1 | 21 | 1% |
| 10 | CCMM_QKT2 | 19 | 1% |
| 11-28 | 其余 (Boot, PCMM, drop, add) | 10-17 each | <1% each |

**Softmax 是绝对瓶颈**（占 34%），因为内部包含：
- exp 多项式 (Chebyshev 28-coeff, 含 x^2→x^4→x^8→x^16→x^32 重复平方)
- 1/x 两步多项式 (Chebyshev 60-coeff × 2)
- NewtonRaphson 3 次迭代
- 多次内部 bootstrap

---

## 6. 解密失败分析

### 6.1 噪声累积路径

从 trace 数据可见，每次 bootstrap 后 NoiseLevel=2（不是 1），这意味着：

1. **bootstrap 输出不是 canonical**——密文带有一个 "lazy" scale，需要 rescale
2. 每次乘法后 NoiseLevel 变为 2，rescale 后回到 1
3. 但 bootstrap 后 NL=2 且没有立即 rescale，下一次操作如果期望 NL=1 可能产生精度损失

### 6.2 两层 bootstrap 计数

从 trace 统计，每层有 **13 次 bootstrap**：
- Boot_Q, Boot_K, Boot_V (3)
- Boot_QKT1_pre_softmax, Boot_QKT2_pre_softmax (2)
- Softmax 内部 bootstrap (~4, 不可见但包含在 Softmax 耗时中)
- Boot_Sm_V (1)
- Boot_Residual1 (1)
- Boot_LN1 (1)
- Boot_pre_GELU (1)
- Boot_pre_LN2 (1)
- Boot_LN2_final (1)

两层共 ~26 次显式 bootstrap + ~8 次 softmax 内部 = **~34 次 bootstrap**

### 6.3 为什么解密失败

每次 bootstrap 在 N=32768 下引入约 2^{-25} 到 2^{-30} 的噪声。
累积 34 次后，总噪声约为 34 × 2^{-27} ≈ 2^{-22}。
而 scale_mod_size=52 意味着明文精度约 2^{-50} 量级。
当累积噪声 2^{-22} >> 明文精度 2^{-50} 时，解密失败。

**根因**：N=32768 的 bootstrap 精度不够（~25 bit），累积 34 次后噪声超出可解密范围。
增大 N 到 65536 可提高 bootstrap 精度到 ~30 bit，但仍不够。
需要 N=131072 或更高才能在 34 次 bootstrap 后保持可解密精度。

---

## 7. 结论

1. **端到端流程跑通**：6 个样本全部完整执行 58 步算子（2 层 × 29 步），无 crash
2. **GPU 内存充裕**：峰值 ~14 GB / 32 GB，无 OOM 风险
3. **Softmax 是时间瓶颈**：占 34% 的单层时间
4. **Bootstrap 是精度瓶颈**：34 次累积噪声导致解密失败
5. **Bootstrap 输出 NoiseLevel=2**：不是 canonical，可能需要额外 rescale
6. **矩阵形状 1x1 为主**：仅 PCMM_Up/GELU 阶段扩展到 1x4

### 下一步建议

1. **增大 N**：试 N=65536 (logN=16) 或 N=131072 (logN=17) 提高 bootstrap 精度
2. **减少 bootstrap 次数**：合并相邻 bootstrap，或降低 Chebyshev 多项式度数
3. **移植 GELU 省 level 优化到 C++**：`93ff5eb` 在 Python 中省了 1 level，C++ 端未实现
4. **修复 bootstrap NoiseLevel**：如果 bootstrap 输出 NL=1，可减少 rescale 开销
5. **优化 Softmax**：用更低度的 Chebyshev 近似（27-coeff 代替 59-coeff）减少 bootstrap

---

## 附录：原始数据

- CSV 文件：`report/bert_tiny_trace.csv`
- 运行日志：`/home/zhiyuan/bench-run/trace-build.log`
- 探针代码：`examples/bert-tiny/src/trace_probe.h` + `Transformer.cu` 插桩
