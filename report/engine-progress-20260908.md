# 引擎侧进度：FIDESlib / OpenFHE 兼容层的改动

日期：2026-09-08。分支 `bootstrap-dev`。基线：上游 `786c760`（FIDESlib 2.1.3）。

这份文档只讲**引擎本身**改了什么——`api/`（OpenFHE 兼容层）、`src/CKKS/`（CUDA 内核）、
`python/src/`（pybind11）。THOR 移植进度在 [`progress-20260908.md`](progress-20260908.md)。

每一项都标了 **算子依赖**（哪个 THOR 原语用它）和 **层依赖**（哪些 stage 会因为它缺失而跑不起来），
因为这些功能不是通用增强，全部是被 THOR 的某个具体需求逼出来的。

相对上游的规模：`api/` +660 行（含两个新文件），`src/CKKS/` +580 行，两个新 example。

---

## 0. 一览

| # | 功能 | 状态 | 关键算子 | 受影响的 stage |
|---|---|---|---|---|
| 1 | GPU bootstrap | ✅ 已通 | `bootstrap` | 07, 13, 15, `he_inv`, `he_invsqrt` |
| 2 | THOR 原语补齐 | ✅ 已通 | `multiply_1j`、`conjugate`、`EvalMultByInteger` 等 | 几乎全部 |
| 3 | 惰性重线性化（degree-2 密文） | ✅ 已通 | `multiply(relin=False)` + `relinearize` | 06, 08, 13, `he_inv` |
| 4 | Light plaintext（T1） | ✅ 远程实测 | `EvalMultLightPt` / `EvalAddLightPt` | 03–05, 10, 12, 14, 17, 18 |
| 5 | Level 截断密钥（T6 的一半） | ✅ 已生效 | `rotate` | 全部含旋转的 stage |
| 6 | `MAXP` 边界检查 | ⚠️ **未编译** | 所有模运算 | 全部（参数选择约束） |
| 7 | pybind11 `pyfideslib` | ✅ 已通 | 全部 | 全部 |
| 8 | 六个已修 bug | ✅ | 见第 8 节 | — |

---

## 1. GPU bootstrap

`EvalBootstrapSetup(levelBudget, dim1, slots, correctionFactor)` / `EvalBootstrapKeyGen`，
配 `CCParams::SetSecretKeyDist(SPARSE_TERNARY)`。上游有 CPU 路径，GPU 侧的 StC/CtS 预计算和
密钥装载是本项目补的（`AddBootstrapPrecomputation`）。

* **算子依赖**：`Stages.bootstrap`
* **层依赖**：**stage 07**（softmax 前的折叠刷新）、**stage 13**（GELU 前）、**stage 15**（FF 残差），
  以及 `he_inv` / `he_invsqrt` 内部的刷新。一层里共 **18 次自举**。
* **现状**：context 能建起来，密钥能生成；**整层从未跑通**，见第 9 节。

## 2. THOR 原语补齐

上游的 OpenFHE 兼容层缺 THOR 需要的几个。新增：

| API | 用途 | 算子依赖 | 层依赖 |
|---|---|---|---|
| `EvalMultByI` | 乘虚数单位（`multMonomial(N/2)`） | `multiply_1j` | 01, 07, 13, 15, 17；`_broadcast_softmax`、LayerNorm 的复数折叠 |
| `EvalConjugate` + `EvalConjugateKeyGen` | 共轭 | `conjugate` | 01, 03–05, 10, 12, 14, 15, 17；所有「取实部」的收尾 |
| `EvalMultByInteger` | 整数乘，**不消耗 level 也不改 scale** | `multiply(x, int)` | `he_inv` 的 `DeltaCiphertext` 整数缩放、`_broadcast_softmax`、GELU 的 ×64 |
| `EvalAddScalar`/`EvalSubScalar`/`EvalMultScalar`/`EvalScalarSub` | 标量四则 | `add`/`subtract`/`multiply` | 多项式求值的常数项、`he_exp` 的中心化、GELU 的 +½ |
| `EvalLevelReduce` | 显式降 level | `level_down` | FIXEDMANUAL 下所有对齐点：`align`、`pcmm` 的未旋转项、残差连接 |
| `GetRemainingLevels` / `GetConsumedLevels` | 读 level | `engine.level` | `align`、`plan_rotation_keys`、明文编码的 level 选择 |

`EvalMultByInteger` 值得单独说：THOR 的 Goldschmidt 除法把「值 = 密文 / delta」这个表示法一路带下去，
每轮用一个整数 `k` 把量级拉回来。整数乘在 CKKS 里是免费的（不动 scale），这是那套算法能在预算内
收敛的前提。

## 3. 惰性重线性化

`EvalMultNoRelin` / `EvalSquareNoRelin` 产生 degree-2 密文，`EvalRelinearize` 在需要时才降回 degree-1。

* **算子依赖**：`multiply(relin=False)`、`square`、`relinearize`
* **层依赖**：**stage 06 / 08**（两个 CCMM 累加大量乘积后只重线性化一次）、
  **stage 13**（GELU 最后那次密文乘）、`he_inv` 的 `_times`、`evaluate_polynomial` 的幂基。
* 省的是密钥交换次数，不是 level。

## 4. Light plaintext（T1）

**这是 THOR 能放进 32 GB 的前提。** 一个 `LightPlaintext` 存 N 个中心化 int64 系数
（`round(Δ·IFFT(m))`），而不是 (L+1) 个 RNS tower：N=2^16 下 **0.5 MiB vs 16 MiB**，约 32 倍。

新增：`api/LightPlaintext.{hpp,cpp}`、`MakeLightPlaintext`、`ExpandLightPlaintext(lp, level)`、
`EvalMultLightPt` / `EvalAddLightPt`、FIFO 展开缓存（`light_plaintext_cache_capacity`，
`ClearLightPlaintextCache`、`GetLightPlaintextCacheSize`）、
CUDA 侧 `RNSPoly::loadCentredCoefficients` + `expandCentredCoeffs_` 内核（逐 limb 有符号取模，
再走现有 NTT）。

* **算子依赖**：`multiply(ct, LightPlaintext)`、`add(ct, LightPlaintext)`、`Stages.plaintext()`
* **层依赖**：所有带权重的 stage——**03/04/05**（QKV）、**10**（attention dense）、
  **12/14**（FF1/FF2）、**17/18**（pooler/classifier），以及所有掩码乘
  （`rotate_internal`、LayerNorm 的 value/statistic、pooler 的 CLS 掩码）。
* **规模**：12 层的权重按满 tower 存要 ~110 GB；light 形式每 stage ≤ 3 GB。
* **状态**：远程 GV100 实测通过。缓存容量默认已从 64 改到 8（每项 `(depth+1)·N·8` ≈ 27 MiB）。

## 5. Level 截断密钥

一把完整旋转密钥是 `2·dnum·(L+1+K)·N·8` 字节——depth 50 / dnum 4 / N=2^16 下 **248 MiB**。
THOR 的旋转键大多只在固定 level 用，所以只存前缀 limb。

新增：`SetRotationKeyLevels(map<index, maxRemainingLevel>)`、
`KeySwitchingKey::{maxLevel, ensureLevel, rebuildAtLevel, coversLevel, describe}`、
`LimbPartition::{resetDecompAndDigit, decompDigitLevelCovered}`、
`ContextData::{truncateKeys, keyLevelMargin, allowKeyGrow}`、
`GetBootstrapKeyLevelPlan`、`printKeyMemoryReport`、`GetKeyDeviceBytes`、`GetGrownKeyCount`。

* **算子依赖**：`rotate`
* **层依赖**：全部含旋转的 stage。旋转计划由 `thorfhe.he.plan_rotation_keys` 计算而非手填——
  跑一遍层，记录每次旋转发生的 level。
* **配套策略**：截断键被用在计划之外的 level 时**默认抛异常**（`allowKeyGrow=false`），
  因为那几乎总是 (delta→level) 表写错了；静默重载会变成每次调用一次重建。
* **实测**：binary rotations 下 15 把键有 15 把被截断，省 3.4 GiB。
  **但对 THOR 这个负载收益有限**——210 把键里 140 把用在接近满 level 的 stage 01–05（第一次自举之前），
  没什么可截。日志里的 `0 truncated` 曾被当成 bug，其实是正确行为。

## 6. `MAXP` 边界检查 ⚠️ 未编译

`src/ConstantsGPU.cuh` 的 `constexpr int MAXP = 64` 是 RNS 素数个数的硬上限，而且它**同时是**
若干扁平指针表的**步长**（`w[i*MAXP + primeid]`）。填表时特殊素数写在 `primes[L+i]`，
所以约束是 `L + K ≤ 64`——**上游没有任何检查**。越界不报错，会依次污染相邻的
Barrett/Shoup 常数、`primeid_flattened`、指针表。

远程报的三个「不同的 bug」（depth≥50 NaN、≥58 segfault、≥63 `cudaMemcpy 'invalid argument'`）
是**同一个溢出越走越远**。`dnum=4` 之后 depth=50 通过，验证了这个诊断。

* **算子依赖**：所有模运算（第一个暴露的是 `EvalMultByI` → `multElement` → `Mult_` 内核）
* **层依赖**：全部。这是参数选择的硬约束，不是某一层的问题。
* **已加**：`SetupConstants` 入口抛带 `q / p / total` 三个数字的异常，并提示
  「降 depth，或升 dnum（K 会变小），或改 MAXP 重编（代价是 `DecompAndModUp_matrix` 的 O(MAXP³)）」。
* **状态**：本机没有 GPU，**这段 C++ 一行都没编译过**，等远程验证。

## 7. pybind11 `pyfideslib`

`python/src/bindings.cpp` + `python/pyfideslib/__init__.py` 的 `Engine`：numpy 进出，
`device="cpu"|"cuda:0"` 切换（`devices.empty()` → OpenFHE CPU 回退），暴露上面全部原语。

* **层依赖**：全部——这是 THOR Python 侧唯一的入口。
* CPU 路径是 GPU 路径的对拍基准（`PYFIDESLIB_DEVICES=cpu` vs `cuda:0` 跑同一份 pytest）。

## 8. 已修的 bug

| bug | 位置 | 影响 |
|---|---|---|
| `KeySwitchingKey::cc` 是悬垂引用 | `KeySwitchingKey.cuh` | `LoadContext` 把局部 Context move 进 `std::any`，引用失效 → `ensureLevel` 段错误 |
| `LightPlaintextImpl::Load` 留下 `uid = 0` | `LightPlaintext.cpp` | 从磁盘读回的**所有权重都别名到展开缓存的第一项**——静默算错 |
| `~CryptoContextImpl` 抹掉 OpenFHE 全局静态密钥表 | `CryptoContext.cpp` | 第二个 context 无法使用密钥 |
| `SetDevices(devices)` 左值 | `CryptoContext.cpp` | 设备列表丢失 → 静默回退到 CPU |
| 截断密钥 grow 不对称 | `LimbPartition.cu` | 改成 reset + 重建，两侧加 `cudaDeviceSynchronize` |
| step3 的 5 个构建/接口 bug | 多处 | 见 git 历史 |

第 2 个值得强调：它不会崩，只会算错，而且只在**从磁盘读多个权重**时才出现——单文件的往返测试
抓不到。修法是把 uid 赋值移进构造函数，让这个 bug 类不可能再出现。

## 9. 当前阻塞（引擎侧）

**整层从未在 GPU 上跑完过。** 两个预算不相交：

| 墙 | 约束 | 来源 |
|---|---|---|
| level | depth ≥ 52 | 一层最深的链 37 层（stage 07 自举 → stage 13 GELU）+ bootstrap 自身约 14 层 |
| 显存 | depth ≤ ~44 | depth 52 预测 35.3 GiB > 32 GiB，还要留 3 GiB 给 keygen 的 decomp 临时 buffer |

`python/thorfhe/budget.py` 按远程日志标定了这个模型（单 key 大小复现到 248.0 MiB，K=11），
`python -m thorfhe.bench budget` 可以在建 context **之前**算出来。

### 引擎侧还能做的

1. **`levelBudget={4,4}`**：唯一未量过、且同时影响两堵墙的杠杆。需要一次实测。
2. **旋转密钥按需加载 / 分批 keygen**：`AddRotationKeys` 一次性生成全部键，`generateAllDecompAndDigit`
   的临时 buffer 是这次 OOM 的直接原因。逐个生成 + 释放能降峰值。
3. **keygen 时不驻留 bootstrap 明文**（9.3 GiB）：那阶段用不到。
4. **种子压缩密钥 `a`**（T6 剩下的一半）：能减半密钥体积，需要改 FIDESlib 的 OpenFHE 补丁分支
   `KeySwitchGen`。
5. **拆开 9 GiB 的 "everything else"**：占预算四分之一还多，目前是标定的黑盒；
   在 keygen 各阶段间打 `nvidia-smi` 就能拆。

## 10. 引擎侧测试

* `test/` 里的上游用例仍然全过。
* `examples/key-truncation/`：两个新 example，`key_truncation`（功能）和 `key_grow_repro`
  （三个场景 `inside` / `above` / `grow`，给 compute-sanitizer 用）。
* `python/tests/`：本机 85 passed / 40 skipped（skip 的是需要编译扩展的），
  远程 GPU 上 111 passed。
* 对拍口径：同一份 stage 代码跑在 `ClearEngine`（numpy 精确算术）和 `pyfideslib.Engine` 上，
  差值只应是 CKKS 噪声。
