# GPU Benchmark Bug: OOM during ExpandLightPlaintext (15 keys + bootstrap)

日期：2026-09-07，commit `e4c0189`（binary rotation decomposition 后）。分支 `bootstrap-dev`。
远程 V100 (Quadro GV100 32GB, sm_70, CUDA 12.9)。

## 现象

使用 `binary_rotations=True`（210 keys → 15 keys）后，key 显存大幅下降，但仍在
`ExpandLightPlaintext` 时 OOM。

参数：`depth=50, dnum=4, log_n=16, bootstrap_level_budget=(3,3), SPARSE_TERNARY`

### 显存占用

```
Plaintexts loaded: 378 ~ 10395MB              (bootstrap plaintexts)
Rotation keys loaded: 49 ~ 12152MB (untruncated estimate)
key memory: 49 rotation keys + eval key, resident 12044 MiB,
            15 truncated (3364 MiB), 0 grown at runtime
```

| 组件 | 占用 |
|------|-----:|
| Bootstrap plaintexts (StC/CtS 矩阵) | 10.4 GB |
| Rotation keys (15 keys, truncated) | 8.7 GB (原 12.4 GB 省了 3.4 GB) |
| Bootstrap keys | ~11.9 GB (含在 key memory 里) |
| **合计 key/plaintext** | **~31 GB** |
| 剩余给密文/light plaintext 展开 | **~1 GB** |

### 崩溃堆栈

```
ExpandLightPlaintext
  → GetExpandedLightPlaintext
    → Plaintext::loadLight
      → RNSPoly::loadCentredCoefficients
        → RNSPoly::grow
          → LimbPartition::generateLimbToLevel
            → LimbPartition::generate
              → Limb ctor → VectorGPU → GPUmalloc
Cuda failure CudaUtils.cu:400: 'out of memory'
```

## 分析

Binary rotation decomposition 成功将 rotation key 从 210 降到 15，key 显存从 50.9 GB 降到 8.7 GB。
但 **bootstrap 的 10.4 GB plaintext + 11.9 GB bootstrap key** 仍然占 ~22 GB，加上 8.7 GB rotation key
= ~31 GB，几乎吃满 32 GB，剩余 ~1 GB 不够做 light plaintext 展开（一次展开需要 ~50 MB，
但 `light_plaintext_cache` 容量 64 项 = ~3.2 GB）。

## 已验证

| 测试 | 结果 |
|------|------|
| 111 pytest (含 GPU, binary_rotations) | 111 passed |
| Engine + bootstrap (无 rotation keys) | OK, 12.4 GB |
| Engine + 15 rotation keys + bootstrap | OK (创建成功), ~31 GB |
| `multiply_1j` at depth=50, dnum=4 | OK, err ~1e-9 |
| Benchmark 运行到 `ExpandLightPlaintext` | **OOM** |

## 建议

1. **减小 `light_plaintext_cache` 容量**：默认 64 项，每项 ~50 MB = 3.2 GB。设为 4 或 8 可省出
   ~2.5 GB。Engine 构造参数已有 `light_plaintext_cache=64`，改为 `light_plaintext_cache=8` 即可。

2. **`levelBudget={4,4}` 替代 `{3,3}`**：RESPONSE 文档提到这会减少 bootstrap 的 giant-step 数，
   明文和 key 一起降。代价是 bootstrap depth 增加。

3. **Bootstrap plaintext 移到按需加载**：378 个 StC/CtS plaintext 不是每次 bootstrap 都全用，
   可以分批加载/卸载。

4. **降低 depth**：`depth=40, bootstrap_level=30` 如果能跑完整层，会减少 L，从而减少每 key 和
   bootstrap 的 limb 数。
