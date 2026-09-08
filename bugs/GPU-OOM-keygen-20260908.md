# GPU Benchmark Bug: OOM during key generation even with binary rotations (depth=44)

日期：2026-09-08。Commit：`a2611ef`。分支 `bootstrap-dev`。
远程 V100 (Quadro GV100 32GB, sm_70, CUDA 12.9)。

## 现象

使用远程建议的 `depth=44, dnum=4, bl=38, binary_rotations, levelBudget=(3,3), light_plaintext_cache=4`，
budget 预测 31.4 GiB（headroom +0.6 GiB），但 key 生成阶段就 OOM。

### Budget 预测 vs 实际

```
predicted GPU footprint (15 rotation keys):
  rotation keys               2.9 GiB  15 keys
  bootstrap keys             10.5 GiB
  bootstrap plaintexts        9.0 GiB
  everything else             9.0 GiB  calibrated from round 3
  total                      31.4 GiB
  headroom on card            0.6 GiB
```

实际日志：
```
Plaintexts loaded: 378 ~ 9261MB
key memory: resident 10844 MiB, 15 truncated (3004 MiB)
```

即 key + plaintext ≈ 20.1 GiB，加上 "everything else" 9 GiB = 29.1 GiB，剩 ~3 GiB。
但 key 生成过程中的 `generateAllDecompAndDigit` 需要额外 GPU 临时内存。

### 崩溃堆栈

```
AddRotationKeys
  → KeySwitchingKey::Initialize
    → RNSPoly::generateDecompAndDigit
      → LimbPartition::generateAllDecompAndDigit
        → LimbPartition::generate
          → Limb ctor → VectorGPU → GPUmalloc
Cuda failure CudaUtils.cu:400: 'out of memory'
```

## 分析

`KeySwitchingKey::Initialize` 在生成 key 时调用 `generateAllDecompAndDigit`，这需要分配
decomp/digit 的临时 GPU buffer。这些 buffer 在 key 生成完成后会释放，但峰值时需要额外 ~2-3 GiB。

budget 模型的 "everything else" 9 GiB 是从 round 3 标定的**稳态**占用（keygen 完成后），
不包含 keygen 过程本身的峰值。keygen 峰值 = 稳态 + 临时 decomp buffer。

## 已试参数

| depth | dnum | bl | budget total | headroom | 结果 |
|------:|-----:|---:|------:|------:|------|
| 50 | 4 | 40 | 34.4 GiB | -2.4 | OOM (ExpandLightPlaintext) |
| 48 | 4 | 38 | 33.2 GiB | -1.2 | OOM (ExpandLightPlaintext) |
| 44 | 4 | 38 | 31.4 GiB | +0.6 | OOM (AddRotationKeys keygen) |
| 44 | 5 | 38 | 34.8 GiB | -2.8 | OOM (未跑，budget 预测) |

## 建议

1. **keygen 分批执行**：`AddRotationKeys` 一次性生成所有 15 个 rotation key，每个 keygen 需要
   decomp/digit 临时 buffer。如果改为逐个生成 + 释放临时 buffer，峰值会降低。

2. **keygen 时释放 bootstrap plaintext**：bootstrap plaintext (9.3 GiB) 在 keygen 阶段不需要，
   可以延迟加载。如果 keygen 时不持有 bootstrap plaintext，剩余 ~11 GiB 足够 keygen 临时 buffer。

3. **`levelBudget={4,4}`**：减少 bootstrap plaintext 数量。需要实测一次才能知道净收益。

4. **降低 depth 到 40 以下**：受限于 `bl≥38`，depth 不能低于 ~52（38 + bootstrap depth ~14）。
   但如果 `levelBudget={4,4}` 能减少 bootstrap depth，可能有空间。
