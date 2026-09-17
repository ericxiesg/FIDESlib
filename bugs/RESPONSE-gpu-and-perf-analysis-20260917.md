# GPU sb=59 跑通（19min）但精度仍炸 + CPU 瓶颈根因分析

日期：2026-09-17。针对 `a534e20`。

---

## 1. sb=59 GPU 端到端结果

```
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
  --refresh-after-dense --binary-rotations \
  --scaling-bits 59 --first-mod-bits 60 \
  --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --device-memory
```

```
total: 1146.60s (19 min)   ← 和 sb=50 的 1137s 一样快！
layer 0: 1131.17s (98.7%)
encode weights: 12.85s
GPU memory peak: 28.02 GiB / 32 GiB
```

**精度仍然炸：**
```
hidden MAE 2.402e+19  max 1.299e+20
encrypted accuracy 0%  (label 翻转)
```

**比 sb=50 更差**（sb=50: MAE=7.5e17, sb=59: MAE=2.4e19）。这和 noise model 预测的
"sb=59 应该足够"矛盾——**设备 bootstrap 的实际误差比 noise model 假设的大。**

---

## 2. sb=59 vs sb=50 对比

| 指标 | sb=50 | sb=59 |
|---|---|---|
| 时间 | 1137s | 1147s |
| GPU 峰值内存 | 25.2 GiB | 28.0 GiB |
| hidden MAE | 7.5e17 | 2.4e19 |
| accuracy | 0% | 0% |

**sb=59 并不比 sb=50 慢**——之前的 4 小时卡顿不是 sb=59 本身的问题。
两次 sb=59 跑（有/无 `--device-memory`）差异巨大，可能是环境因素。

**sb=59 精度更差**可能是因为 limb 更多（38 vs 33），每次操作的舍入误差累积更多。

---

## 3. 逐 stage 内存 trace

```
stage                   pool GiB   in_use GiB   reclaimable GiB   driver free GiB
(after key gen)          17.02       16.85          0.17             14.09
rotated                  24.02       23.43          0.59              7.07
query                    25.02       23.96          1.06              6.07
key                      25.02       24.24          0.78              6.07
value                    25.02       24.52          0.50              6.07
scores                   25.02       19.78          5.23              6.07
softmax                  28.02       23.93          4.09              3.07  ← 峰值
context                  28.02       18.61          9.40              3.07
attention_dense          28.02       19.62          8.40              3.07
refreshed_dense          28.02       19.18          8.84              3.07
norm_1                   28.02       19.24          8.78              3.07
intermediate             28.02       18.22          9.79              3.07
gelu                     28.02       18.87          9.15              3.07
output_dense             28.02       18.80          9.22              3.07
norm_2_input             28.02       18.90          9.12              3.07
norm_2                   28.02       18.52          9.50              3.07
```

**pool 在 softmax 阶段涨到 28 GiB，driver free 降到 3 GiB**。
之后 in_use 稳定在 18-20 GiB，reclaimable 8-9 GiB——pool 不再增长。

**协作者 §5 的假设部分正确**：driver free 确实很低（3-6 GiB），但 pool 在 softmax
之后不再增长，所以 cudaMalloc 同步不是全程问题——主要影响 softmax 阶段。

---

## 4. 之前的 4 小时卡顿不是 sb=59 的问题

| 跑次 | 参数 | 时间 | 备注 |
|---|---|---|---|
| sb=50 无 device-memory | 默认 | 19 min | 精度炸 |
| sb=59 无 device-memory | 默认 | 4h+ 未完 | 卡死，kill |
| sb=55 无 device-memory | 默认 | 2h+ 未完 | 卡死，kill |
| sb=59 **有** device-memory | 默认 | **19 min** | 精度炸 |

**唯一差异是 `--device-memory`。** 加了 `--device-memory` 后 sb=59 和 sb=50 一样快。
可能的解释：
- `--device-memory` 改变了内存分配策略（probe 回调可能触发了 pool trim）
- 之前的卡顿是某个 CUDA driver 状态问题，重跑后消失
- 环境因素（服务器重启后第一次跑 vs 后续跑）

**结论：sb=59 本身不是性能瓶颈。** 之前的 4 小时是偶发问题。

---

## 5. GPU util 0-5% 的根因分析

### 5.1 内核太小，CPU dispatch 主导

N=65536（2^16），每个 NTT/mod-up/mod-down kernel 在 GV100 上只需微秒级。
但每次操作需要：Python 调用 → pybind11 → C++ → CUDA kernel launch。
**CPU dispatch 时间 >> GPU kernel 时间**，GPU 大部分时间在等 CPU 发下一个 kernel。

### 5.2 pybind11 没有释放 GIL

`bindings.cpp` 中所有 `EvalRotate`、`EvalMult`、`EvalBootstrap` 等 binding
**没有 `py::gil_scoped_release`**。Python GIL 在整个 C++ 调用期间被持有，
无法在等待 GPU 时让其他 Python 线程工作。

### 5.3 binary rotations 放大 8x

`stages.py:105-108`：每个旋转被拆成最多 8 个幂等旋转，每个都是独立的
Python→C++ 调用。一层有 ~1802 个原始旋转 → ~8138 个实际旋转。
**8138 次 Python→C++ round trip，每次只做微秒级 GPU 工作。**

### 5.4 每次 key-switch 创建/销毁 CUDA event

`LimbPartitionMGPU.cu:684,1872`：每次 key-switch 调用
`cudaEventCreateWithFlags` + `cudaEventDestroy`。一层有几千次 key-switch
（8138 次旋转 + bootstrap 内部），每次都有 event 创建/销毁开销。

### 5.5 multScalar 每次分配/释放 GPU 内存

`LimbPartition.cu:2288-2341`：`multScalar` 每次调用
`cudaMallocAsync` + `cudaFreeAsync`。bootstrap 内部有几十次 multScalar。

### 5.6 bootstrap 后的 Python rescale 循环

`pyfideslib/__init__.py:261-270`：bootstrap 后最多 8 次
`GetNoiseLevel` + `Rescale` 的 Python→C++ 调用。17 次 bootstrap = ~272 次额外 round trip。

---

## 6. 优化方向（按影响排序）

### 高影响

1. **批量 binary rotations**：C++ 端加 `EvalRotateMulti(ct, steps[])`，
   一次调用完成所有幂等旋转。消除 7/8 的 Python→C++ round trip。
   预计加速 2-4x。

2. **释放 GIL**：bindings.cpp 中所有 EvalXxx 加 `py::gil_scoped_release`。
   允许 Python 层并行调度多个 GPU 操作。

3. **预分配 CUDA event pool**：替代每次 key-switch 的 create/destroy。
   消除 `cudaEventCreateWithFlags`/`cudaEventDestroy` 的 driver overhead。

4. **预分配 multScalar scratch buffer**：替代每次 `cudaMallocAsync`/`cudaFreeAsync`。

### 中影响

5. **bootstrap rescale 移入 C++**：把 `__init__.py:261-270` 的 rescale 循环
   挪到 `EvalBootstrap` 的 C++ 实现里，消除 ~272 次 round trip/层。

6. **CUDA graph 缓存优化**：graph cache key 为 `(level, moddown)`，
   bootstrap 中很多不同 level 导致 cache miss。改为按 digit 结构 keying。

7. **减少 `trim_auxiliary_polys` 频率**：当前每个 stage 后调用（~16 次/层），
   改为只在 bootstrap 边界调用。

### 需要进一步调查

8. **`--device-memory` 为什么让 sb=59 从 4h 变成 19min**：
   需要对比有/无 `--device-memory` 的 CUDA driver 行为。
   可能是 probe 回调触发了内存 pool 的不同行为。

9. **sb=59 精度比 sb=50 更差**：noise model 假设 sb=59 够，
   但实际 MAE=2.4e19 比 sb=50 的 7.5e17 还差。需要调查 bootstrap 实际误差。

10. **`OMP_NUM_THREADS`**：OpenFHE 的 OpenMP 可能占用过多 CPU 核，
     和 CUDA driver 线程竞争。尝试 `OMP_NUM_THREADS=4`。

---

## 7. 精度问题

**两个精度问题需要分开处理：**

1. **sb=50 和 sb=59 都炸**：说明 bootstrap 精度不够不是唯一的精度问题。
   noise model 在 layer 0 上预测 sb=59 够（max error = 0.00709），
   但 GPU 实测 MAE=2.4e19。**差距 20 个数量级。**

2. **可能的非 bootstrap 精度问题**：
   - rescale 累积误差（FIXEDMANUAL 下每次 rescale 有舍入）
   - key-switch 误差（truncated keys 引入额外噪声）
   - 多项式近似误差（he_exp、he_inv 的 Chebyshev 拟合）
   - 这些在 ClearEngine 上是精确的，只在 GPU 上才暴露

**建议：跑 `--per-stage` 看哪个 stage 开始发散。**
