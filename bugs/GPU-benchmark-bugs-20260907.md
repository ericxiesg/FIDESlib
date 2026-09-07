# GPU Benchmark Bugs — 2026-09-07

分支 `bootstrap-dev`，commit `3a386ee`。远程 V100 (Quadro GV100 32GB, sm_70, CUDA 12.9)。
所有 bug 均在 `--engine fideslib`（GPU 路径）上发现；CPU 路径（`--engine clear`）全部通过。

环境：`log_n=16, scaling_bits=50, first_mod_bits=55, dnum=3`，旋转密钥由 `plan_rotation_keys`
覆盖完整 layer（207 个 index）。

---

## Bug 1: `EvalMultByI` / `multMonomial` — depth ≥ 50 产生 NaN，depth ≥ 58 segfault

**严重度：阻塞**

### 现象

`multiply_1j`（即 `EvalMultByI` → `Ciphertext::multMonomial(N/2)`）在 `log_n=13` 或 `log_n=16` 下：

| depth | first_mod_bits | 结果 |
|------:|:--------------:|------|
| ≤ 48  | 55 | OK，err ~1e-10 |
| 49    | 55 | OK |
| ≥ 50  | 55 | **NaN**（无异常，decrypt 结果全 NaN） |
| ≥ 58  | 55 | **Segfault**（CUDA illegal memory access） |
| ≥ 45  | 60 | **Segfault**（更早崩溃） |
| ≥ 63  | 任意 | Engine 创建即崩溃 `ConstantsGPU.cu:613: 'invalid argument'` |

### 复现

```python
import numpy as np, pyfideslib
for depth in [40, 48, 50, 58]:
    e = pyfideslib.Engine("cuda:0", log_n=13, depth=depth,
                          scaling_bits=50, first_mod_bits=55, dnum=3)
    z = np.random.randn(1<<12) + 1j*np.random.randn(1<<12)
    cz = e.encrypt(z)
    iz = e.multiply_1j(cz)
    err = np.max(np.abs(e.decrypt(iz) - 1j*z))
    print(f"depth={depth}: err={err}")  # depth=50 -> nan, depth=58 -> segfault
    del e
```

### 堆栈（depth=58 segfault）

```
Ciphertext::multMonomial(int power)
  → RNSPoly::multElement(monomial)         // RNSPoly.cpp:543
    → LimbPartition::multElement(p)         // LimbPartition.cu:648
      → s.wait(p.getS())                    // Stream::wait
        → CUDA error: 'an illegal memory access was encountered'
```

### 分析

`multMonomial`（`src/CKKS/Ciphertext.cpp:1758`）流程：
1. `RNSPoly monomial(cc.getAuxilarPoly())` — 从 auxPoly 池取一个 RNSPoly
2. `monomial.grow(c0.getLevel())` — 扩展到目标 level
3. `monomial.dropToLevel(c0.getLevel())` — 确保 level 一致
4. 加载系数 `coefs[power] = 1`，做 NTT
5. `c0.multElement(monomial)` 和 `c1.multElement(monomial)` ← **崩溃点**

`LimbPartition::multElement`（`LimbPartition.cu:648`）用 `getLimbSize(*level)` 计算 limbsize，
然后 `Mult_<<<>>>` kernel 做 element-wise 乘法。`p.limbptr.data + i` 访问 monomial 的 limb 指针。

**怀疑方向**：
- `getAuxilarPoly()` 返回的 RNSPoly 在高 depth 下 limb 分配不完整（`grow` 或 `dropToLevel` 逻辑
  在 depth ≥ 50 时有 off-by-one 或 buffer 未初始化）
- `monomialCache` 的 level 检查 `cc.precom.monomialCache.find(power)->second.getLevel() != this->getLevel()`
  在首次调用时 cache 为空，走入构建路径；构建路径中 `grow` + `dropToLevel` 的组合在高 depth 下
  可能留下未分配的 limb slot，`Mult_` kernel 访问到野指针

### 影响

THOR 的 `stage_01_complexify_x`、softmax、layernorm、feedforward、pooler 全部调用 `multiply_1j`，
这是 BERT 加密推理的核心原语。depth ≤ 48 时正常，但 BERT 一层需要 ~30 levels，depth=48 不够跑完
完整 layer（需要 bootstrap 来刷新 level）。

---

## Bug 2: GPU Bootstrap 运行时崩溃 — stage_02 segfault

**严重度：阻塞**

### 现象

启用 `bootstrap_level_budget=(3, 3)` + `secret_key_dist=SPARSE_TERNARY`，engine 创建成功
（GPU 内存 13.9 GB / 32 GB），但运行 `EncoderLayer.forward` 时在 stage_02
（`make_rotated_copies`，即 `rotate_internal` → `multiply` + `rotate`）segfault。

### 复现

```python
e = pyfideslib.Engine("cuda:0", log_n=16, depth=30, scaling_bits=50,
                      first_mod_bits=55, dnum=3,
                      rotation_indexes={...},  # 207 keys from plan_rotation_keys
                      bootstrap_level_budget=(3, 3),
                      secret_key_dist=pyfideslib.SPARSE_TERNARY)
# engine 创建成功，GPU 内存 13.9 GB
# 运行 layer.forward → stage_01 OK → stage_02 segfault
```

### 堆栈

```
EncoderLayer.forward → stage_02_make_rotated_copies
  → Stages.rotate_internal → multiply(complements[delta], x)
    → Engine.multiply → EvalMultLightPt / EvalMultPt
      → [native segfault, no Python traceback]
```

### 分析

- Engine 创建日志：`Plaintexts loaded: 378 ~ 6426MB`，`Rotation keys loaded: 49 ~ 6026MB`，
  bootstrap key level plan 48 keys, 104 indexes total
- 总 GPU 内存 13.9 GB，剩余 ~18 GB 给密文，不太可能是 OOM
- stage_01（`complexify_x`，含 `multiply_1j` + `conjugate` + `add`）成功
- stage_02（`rotate_internal`，含 `multiply` + `rotate` + `add` + `rescale`）崩溃
- **怀疑方向**：bootstrap key gen 改变了 rotation key 的内部状态或 level plan，导致 stage_02
  的 `EvalRotate` 或 `EvalMult` 访问到不一致的 key 数据

### 影响

没有 bootstrap，BERT 一层消耗 ~30 levels，depth=30 不够。必须修好 bootstrap 才能在 GPU 上
跑通完整 layer。

---

## Bug 3: GPU OOM — depth=48, log_n=16, 无 bootstrap

**严重度：阻塞**

### 现象

`depth=48, log_n=16, first_mod_bits=55`（Bug 1 的上限），不启用 bootstrap，运行
`EncoderLayer.forward` 时在 stage_05（`apply_qkv_weight_bias` → `pcmm`）CUDA OOM。

### 堆栈

```
Cuda failure /home/zhiyuan/workspace/THOR-v2/FIDESlib/src/CudaUtils.cu:400: 'out of memory'
  → RNSPoly::copy → Ciphertext::copy → CopyDeviceCiphertext → CiphertextImpl copy ctor
```

### 分析

- `pcmm`（`stages.py:206`）调用 `parallel_diagonal_pc_mult`（`stages.py:175`），同时持有
  ~144 个临时密文（`out_dim=12, diag_dim=12, pack=16`）
- 每个 RNSPoly 在 depth=48, log_n=16 下约 50 limbs × 65536 coefficients × 8 bytes ≈ 25 MB
- 144 × 25 MB ≈ 3.6 GB 仅密文本体，加上 copy 临时量和 rotation key 约 10 GB，接近 32 GB 上限
- `Ciphertext::copy`（copy ctor）在 `CopyDeviceCiphertext` 中分配新 GPU buffer 时 OOM

### 当前 workaround

降到 `depth=30` 可避免 OOM，但 depth=30 不够跑完完整 layer（需要 bootstrap，见 Bug 2）。

### 建议

- 在 `parallel_diagonal_pc_mult` 中流式处理而非全量持有 144 个密文
- 或在 C++ 侧实现密文池复用（`returnAuxilarPoly` 已存在但未被调用）

---

## Bug 4: `pyfideslib.Engine.add` / `subtract` 不支持 numpy array 参数

**严重度：已修复（本地），需提交者审查**

### 现象

`ClearEngine.add(x, y)` 接受 numpy array 作为 `y`（plaintext operand），但 `pyfideslib.Engine.add`
没有处理 `np.ndarray` 分支，直接走 `EvalAdd(x, y)` → C++ 侧类型不匹配崩溃。

### 修复（已在 `python/pyfideslib/__init__.py`）

```python
def add(self, x, y):
    if isinstance(y, (int, float)):
        return self.cc.EvalAddScalar(x, float(y))
    if isinstance(y, np.ndarray):          # <-- 新增
        pt = self.encode(y, level=self.depth - self.level(x))
        return self.cc.EvalAddPt(x, pt)
    ...

def subtract(self, x, y):
    ...
    if isinstance(y, np.ndarray):          # <-- 新增
        pt = self.encode(y, level=self.depth - self.level(x))
        return self.cc.EvalSubPt(x, pt)
    ...
```

`multiply` 已有 `np.ndarray` 分支（`__init__.py:146`），`add`/`subtract` 遗漏了。

---

## Bug 5: `pyfideslib.Engine.rotate(0)` 查找不存在的 rotation key

**严重度：已修复（本地），需提交者审查**

### 现象

`rotate(x, 0)` 调用 `EvalRotate(x, 0)`，C++ 侧查找 rotation index 0 的 key，不存在时报
`RuntimeError: Rotation index0 not found`。rotation by 0 是 no-op，不应需要 key。

### 修复（已在 `python/pyfideslib/__init__.py`）

```python
def rotate(self, x, delta: int):
    if int(delta) == 0:
        return x
    return self.cc.EvalRotate(x, int(delta))
```

---

## Bug 6: `plan_rotation_keys` 只覆盖 stage 01-05

**严重度：已修复（本地），需提交者审查**

### 现象

`thorfhe/he.py:plan_rotation_keys` 只跑 stage 01-05 来收集 rotation index，但完整 layer
（stage 06-16）需要 207 个 rotation index（vs stage 01-05 的 11 个）。GPU 运行到 stage 06+ 时
因缺少 rotation key 崩溃。

### 修复（已在 `python/thorfhe/he.py`）

改为跑完整 `EncoderLayer.forward`（用 dummy zeros 权重），收集所有 `engine.rotation_levels`，
过滤掉 level < 0 的无效条目。

---

## 本地修改清单

| 文件 | 改动 |
|------|------|
| `python/pyfideslib/__init__.py` | `add`/`subtract` 加 `np.ndarray` 分支（Bug 4）；`rotate(0)` no-op（Bug 5） |
| `python/thorfhe/he.py` | `plan_rotation_keys` 覆盖完整 layer（Bug 6） |
| `python/thorfhe/layer.py` | 删除临时 debug print |

**未修改 C++ 代码** — Bug 1/2/3 需要 C++ 侧修复。

---

## 测试环境

```
GPU: Quadro GV100 (sm_70, 32GB)
CUDA: 12.9 (/usr/local/cuda-12.9)
OpenFHE: 1.5.1.1 (/home/zhiyuan/workspace/THOR-FIDE/openfhe-install)
Python: 3.11
LD_LIBRARY_PATH=/home/zhiyuan/workspace/THOR-FIDE/openfhe-install/lib:/usr/local/cuda-12.9/lib64
PYTHONPATH=/home/zhiyuan/bench-run:/home/zhiyuan/workspace/THOR-v2/FIDESlib/python
```

## CPU benchmark 结果（已跑通，作为对照）

```
engine=clear, layers=1, limit=4, depth=90
accuracy: 100% (4/4)
per-stage fidelity: all passing (query relRMSE 2.9e-7, softmax 2.3e-6, ...)
total: 35.87s
```
