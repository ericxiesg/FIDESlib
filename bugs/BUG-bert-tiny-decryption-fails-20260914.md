# BERT-Tiny 端到端：解密全部失败 — approximation error too high

日期：2026-09-14。环境：Quadro GV100 32GB (sm_70), CUDA 12.9, OpenFHE 1.5.1。

## 现象

`examples/bert-tiny/main/berttiny-all` 完整跑完两层 encoder + classifier（不 crash），
但**每个样本**在 classifier 解密时都报：

```
ckkspackedencoding.cpp:l.453:Decode(): The decryption failed because
the approximation error is too high. Check the parameters.
```

输出永远是 `-1`（解密失败标记），accuracy 0/3。

## 测试矩阵

用官方参数和几种变量组合，均在 GV100 上跑完 10 个 pretokenized 样本（3 warmup + 7 measured）：

| # | 分支/库 | L | N (logN) | KEY_GROW | 是否 crash | 结果 | 时间/样本 | GPU 内存 |
|---|---|---|---|---|---|---|---|---|
| 1 | bootstrap-dev (809b0f9) | 23 | 32768 (15) | 1 | **Yes** — level 下溢 | 0/0 | — | ~5 GB |
| 2 | bootstrap-dev (2b1a029, 含 GELU 省 level + degree 检查) | 23 | 32768 (15) | 1 | **Yes** — level 下溢 | 0/0 | — | ~5 GB |
| 3 | main (fa97286, 含 multPt top-limb 修复) | 23 | 32768 (15) | 1 | No | 0/3 解密失败 | 8.3 s | ~5 GB |
| 4 | bootstrap-dev (2b1a029) | 25 | 32768 (15) | 1 | No | 0/3 解密失败 | 8.6 s | ~5.4 GB |
| 5 | bootstrap-dev (2b1a029) | 25 | 65536 (16) | 1 | No | 0/3 解密失败 | 11.4 s | ~11 GB |

所有组合的 `utils.cu` 其他参数均为官方默认：`scale_mod_size=52, first_mod=56, dnum=1, num_large_digits=3, FLEXIBLEAUTO`。

## 两个不同的故障模式

### 模式 A：level 下溢 crash（#1, #2）

```
[EXCEPTION] vector::_M_range_check: __n (which is 4294967295) >= this->size() (which is 24)
```

`4294967295 = (uint32_t)-1`，密文 level 被降到 -1，访问 `RNSLimbs[level]` 越界。

**只在 bootstrap-dev 库上发生。** main 分支的 `3ece51b`（multPt top-limb fix）修复了它——
同一参数 (#3) 在 main 库上不 crash，说明 crash 是 C++ 库 bug 而非参数深度不足。

### 模式 B：解密失败，不 crash（#3, #4, #5）

程序完整跑完两层 encoder + classifier，但 classifier 解密时噪声超出可解密范围。
**所有库和参数组合都失败。** 增大 N (32768->65536) 和增大 L (23->25) 都不改善。

## 运行时数据（#4: L=25, N=32768, bootstrap-dev 2b1a029）

```
[FIDESlib] bootstrap diagonals: CtS layer 0 holds 25 limbs; a ciphertext at L=25 has 26
[FIDESlib] bootstrap key level plan: 46 of 61 keys truncated (L=25, bootstrap depth 20,
           levelBudget {3,3}, StC starts at 8, margin 1); 104 indexes, by level: 7x7 8x7 24x32
[FIDESlib] key memory: 62 rotation keys + eval key, resident 2823 MiB, 29 truncated (1089 MiB)
Plaintexts loaded: 378 ~ 2551 MB
```

每层 encoder 包含约 17-21 次 bootstrap（QKV bootstrap x3, softmax 内部 exp+1/x poly, GELU, LayerNorm 等），
两层合计约 34-42 次。每次 bootstrap 引入的 CKKS 噪声在 N=32768 下约为 2^{-30} 量级，
累积 40 次后到 classifier 解密时已超出 `scale_mod_size=52` 的可解密范围。

## 关键发现

1. **bootstrap-dev 库的 multPt bug 导致 L=23 crash**——main 分支已修复（`3ece51b`），bootstrap-dev 没有。
   这不是参数深度问题，是 C++ 库 bug。

2. **multPt 修复只防止 crash，不解决噪声**——#3 在 main 库上不 crash 但仍然解密失败。

3. **增大 L 反而更差**——L=25 比 L=23 多 2 个 prime limb，bootstrap 后的初始噪声更大。
   增大 N 到 65536 也不改善（#5），说明瓶颈不是 bootstrap 精度而是**累积噪声次数**。

4. **GELU 省 level 的优化（`93ff5eb`）在 Python thorfhe 中**，bert-tiny C++ 端没有对应的改动——
   C++ `PolyApprox.cu` 中的 `evalGelu` 仍然用 `multScalar(GetPreScaleFactor)` 做输入缩放，
   每次消耗一个 level。如果把这个优化移植到 C++，可以省掉每层 GELU 的一次 rescale，
   减少一层噪声。

5. **原始作者可能在 A100 (sm_80) 上测试**——sm_70 和 sm_80 的浮点累加精度可能不同，
   影响 NTT 和 bootstrap 的数值行为。未确认。

## 已排除的原因

- **不是权重文件问题**——权重和 pretokenized 样本由 `save_weights.py` / `PreTokenize.py`
  从 `prajjwal1/bert-tiny` + SST-2 正确生成。
- **不是 key truncation 问题**——设了 `FIDESLIB_KEY_GROW=1`，warning 日志显示 key 按需 reload 成功。
- **不是 GPU OOM**——最大 ~11 GB（#5），GV100 32 GB 充裕。
- **不是 CUDA 版本问题**——统一用 CUDA 12.9 + sm_70，fideslib.a 和 berttiny-all 编译一致。

## 建议的下一步

1. **把 main 分支的 multPt top-limb 修复 cherry-pick 到 bootstrap-dev**——消除 #1/#2 的 crash，
   让 L=23 可以完整跑完（虽然 #3 表明跑完也解密失败，但至少能拿到中间层 noise 数据）。

2. **把 GELU 省 level 的优化移植到 C++ `PolyApprox.cu`**——`93ff5eb` 在 Python `numeric.py` 中
   把 `1/carrier` 除法折入多项式系数，省掉一次 `rescale(multiply(x, 1/carrier))`。
   C++ 端的 `evalGelu` 有同样的 `multScalar(GetPreScaleFactor)` 调用，可以同样优化。

3. **在 classifier 前加一次 bootstrap**——目前 classifier 解密时的密文已经过了两层 encoder 的
   ~40 次 bootstrap，可能需要最后一次 bootstrap 刷新噪声再解密。

4. **联系 FIDESlib 作者确认**——这个 benchmark 是否在 GV100/sm_70 上跑通过？
   官方测试的 GPU 型号和 CUDA 版本是什么？

5. **减少 bootstrap 次数**——如果能合并相邻的 bootstrap（如 softmax 内的 exp^32 -> 1/x 两步之间），
   或降低 Chebyshev 多项式度数（用 27-coeff 代替 59-coeff），可以减少累积噪声。

---

## 复现步骤

```bash
# 环境
export FIDESLIB_NUM_GPUS=1
export FIDESLIB_RING_DIM=15   # N=32768
export FIDESLIB_KEY_GROW=1
export LD_LIBRARY_PATH=<openfhe>/lib:/usr/local/cuda-12.9/lib64

# 权重和 pretokenized 样本已生成在
# examples/bert-tiny/weights/weights-bert-tiny-sst2/

# 编译
cd examples/bert-tiny/build
cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_PATH=/usr/local/cuda-12.9 -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCMAKE_CUDA_ARCHITECTURES "70-real" \
  -Dfideslib_DIR=<fideslib-install>/share/fideslib/cmake
make berttiny-all -j4

# 运行
./berttiny-all
# -> 所有样本输出 "Decryption failed: approximation error is too high"
# -> Final Accuracy: 0/3
```

## 参数配置详情

完整的官方参数、矩阵 layout、密文/密钥 level 配置、两层 encoder 流程图见
`report/bert-tiny-official-params-notes.md`（本地，未 push）。
