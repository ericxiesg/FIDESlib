# 干净重编译修了崩溃；07d 和 11b 的发散和旧 install 一致

日期：2026-09-21。针对 `9eef359`。**崩溃已修复，探针数据如下。**

---

## 0. 你说对了：是工具链，不是源码

按你的步骤 1，全量干净重编译，CUDA 钉死 12.9：

```bash
rm -rf build build-py fideslib-install
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_PATH=/usr/local/cuda-12.9 \
      -DFIDESLIB_ARCH=70-real \
      -DOpenFHE_DIR=.../openfhe-install/lib/OpenFHE \
      -DFIDESLIB_INSTALL_PREFIX=.../fideslib-install
cmake --build build -j && cmake --install build

cmake -S python -B build-py \
      -DCMAKE_PREFIX_PATH=.../fideslib-install \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.9
cmake --build build-py
```

**崩溃消失了。** 不需要步骤 2（GIL 开关）。

根因确认：
1. `/usr/local/cuda` 从 12.9 换成了 13.3（CUDA 13 删了 sm_70）
2. cmake 不追踪 header 依赖 → 增量构建跨 `Bootstrap.cuh` 签名变化 → ODR 违反
3. `build-py` 的 `find_package(CUDAToolkit)` 默认走 `/usr/local/cuda`（13.3），和 12.9 编的 fideslib.a 链接出 `undefined symbol: cudaStreamGetCaptureInfo_v2`

---

## 1. 探针数据（clean build，`--inverse-lift 3`，1 layer，1 sample，268.6s 冷跑）

### softmax（stage 07）

```
07a.refreshed_scores    8 ct  level 17  min -16.05     max +37.5        p50 1.015     p99 6.592     <0 119510/262144  >1 76927/262144
07b.exp                 8 ct  level  9  min -1.412e-09 max +0.1135      p50 2.182e-13 p99 0.002666  <0 86546/262144   >1 0/262144
07c.denominator         1 ct  level  9  min -1.058e-09 max +0.3092      p50 9.885e-12 p99 0.1763    <0 12080/32768    >1 0/32768
07d.inverse_denominator 1 ct  level 16  min -0.008294  max +5.715e+05   p50 2.991e-10 p99 0.03814   <0 12255/32768    >1 1/32768
```

### layernorm（stage 11）—— 第一组（stage 11 的 LN1）

```
11.residual             8 ct  level 20  min -9.439e+04  max +9.141e+04   p50 1.409e+04 p99 5.356e+04
11a1.n_sum_of_squares   1 ct  level 17  min -1.149e-07  max +8.561e+07   p50 3.338e-08 p99 7.953e+07  <0 14536/32768  >1 2048/32768
11a2.squared_total      1 ct  level 17  min -3.207e-11  max +1.167e+06   p50 3.198e-12 p99 1.549e+05  <0 16860/32768  >1 2048/32768
11a.variance            1 ct  level 17  min -1.164e-07  max +8.559e+07   p50 3.321e-08 p99 7.943e+07  <0 15511/32768  >1 2048/32768
11b.inverse_sqrt        1 ct  level 10  min -2.153e+154 max +2.171e+154  p50 3.7e+153  p99 1.414e+154 <0 16543/32768  >1 16225/32768
```

### layernorm —— 第二组（stage 11 的 LN2，`--refresh-after-dense` 后）

```
11a1.n_sum_of_squares   1 ct  level 14  min -1.094e-05  max +7.999e+09   p50 3.045e-06 p99 7.349e+09  <0 14336/32768  >1 2048/32768
11a2.squared_total      1 ct  level 14  min -2.754e-09  max +9.751e+07   p50 2.309e-10 p99 1.739e+07  <0 14921/32768  >1 2048/32768
11a.variance            1 ct  level 14  min -1.105e-05  max +7.997e+09   p50 3.21e-06  p99 7.335e+09  <0 14336/32768  >1 2048/32768
11b.inverse_sqrt        1 ct  level  7  min -1.532e+109 max +1.537e+109  p50 2.604e+108 p99 9.783e+108 <0 16415/32768  >1 16353/32768
```

---

## 2. 和旧 install 对比

| 探针 | 旧 install（`8fee326`，Sep 16） | clean build（CUDA 12.9） |
|---|---|---|
| 07d max | 7.326e+05 | 5.715e+05 |
| 07d <0 | 12410/32768 | 12255/32768 |
| 07d >1 | 1/32768 | 1/32768 |
| 11b max | 2.084e+154 | 2.171e+154 |
| 11b <0 | 16503/32768 | 16543/32768 |
| 11b >1 | 16265/32768 | 16225/32768 |

**几乎完全一致。** 这确认了：
- 旧 install 的 `--inverse-lift 3` 结果是有效的
- 发散不是构建环境问题，是算法层面的

---

## 3. 你要的 `inv_iter01_b` 到 `inv_iter03_b`

**这次跑没带 `THORFHE_DEBUG=1`**（我先确认崩溃修了再跑 debug 版）。
上面的探针是 `--per-stage` 的，不是 he_inv 逐迭代探针。

**下一步**：带 `THORFHE_DEBUG=1 --per-stage --inverse-lift 3` 用热 cache 重跑
（~90s），把 `inv_iter01_b` 到 `inv_iter03_b` 的 `<0`/`>1` 计数给你。

---

## 4. per-stage fidelity

```
query      scale  2.0000  relRMSE 2.884e-07   ✅
value      scale  2.0000  relRMSE 3.141e-07   ✅
scores     scale  0.0313  relRMSE 2.874e-07   ✅
softmax    scale  5.4e+76 relRMSE 3.682e+02   ❌ (07d 发散)
attention  scale -6.2e+16 relRMSE 1.626e+02   ❌
norm_1     scale -2.1e+96 relRMSE 1.594e+03   ❌ (11b 发散)
```

query/value/scores 完全正确（relRMSE ~3e-07）。从 softmax 开始爆。

---

## 5. 现在的状态

- **构建**：CUDA 12.9 干净全量重编译，fideslib.a + .so 都 OK
- **plaintext cache**：401 fields loaded, 290 encoded（冷跑刚建的，下次热跑 ~90s）
- **崩溃**：已修复
- **accuracy**：encrypted 0%（1 sample 分错）
- **主要问题**：07d（softmax 逆元）和 11b（LayerNorm 逆平方根）仍发散

---

## 附：构建命令（完整，可复现）

```bash
# fideslib
rm -rf build fideslib-install
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_PATH=/usr/local/cuda-12.9 \
  -DFIDESLIB_ARCH=70-real \
  -DOpenFHE_DIR=/home/zhiyuan/workspace/THOR-FIDE/openfhe-install/lib/OpenFHE \
  -DFIDESLIB_INSTALL_PREFIX=/home/zhiyuan/workspace/THOR-v2/fideslib-install \
  -DCMAKE_CUDA_FLAGS='-DNDEBUG -UNDEBUG' \
  -DCMAKE_CXX_FLAGS='-DNDEBUG -UNDEBUG'
cmake --build build -j
cmake --install build

# Python bindings
rm -rf build-py
cmake -S python -B build-py \
  -DCMAKE_PREFIX_PATH=/home/zhiyuan/workspace/THOR-v2/fideslib-install \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9
cmake --build build-py

# Run
PYTHONPATH=.../python python3 -m thorfhe.bench fhe \
  --engine fideslib --device cuda:0 --depth 37 --dnum 4 \
  --bootstrap-level-budget 3,3 --binary-rotations --refresh-after-dense \
  --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --layers 1 --limit 1 --per-stage --compact \
  --extra-rotation-keys 6 --rotation-max-steps 4 \
  --inverse-lift 3 --plaintext-cache /home/zhiyuan/ptcache
```
