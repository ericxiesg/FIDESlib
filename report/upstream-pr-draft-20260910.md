# FIDESlib 上游 PR 草稿

日期：2026-09-10，2026-09-11 更新基线。分支 `bootstrap-dev`。
基线：上游 `main` = `fa97286`（PR #38、#39 合入之后）。

**这份文档只覆盖引擎侧改动。** THOR 移植不进 PR。第一到第四节里不出现 THOR 的算法细节，
所有与 THOR 相关的内容集中在第五节；第三节列出**目前还残留在引擎代码里、提交前必须清掉的
THOR 字样**。

正文为纯文本，不含零宽字符、不可见标记或任何形式的隐藏水印；后续编辑请保持这一点。

> **2026-09-11 更新**：已按本草稿把代码实际切开，放在 `../patch/PR1..PR8/`（完整文件，非 diff），
> 说明见 `../patch/README.md`。切分过程验证了两件事：八个 PR 全部应用回去与工作树**逐字节相同**，
> 且没有任何 PR 的树引用**更靠后的 PR 才引入的符号**。切的过程也纠正了本草稿三处，已就地改掉：
> PR2 是**七个**修复不是五个（多出 `EvalSub` 符号反了、`ConstPlaintext` 的 `any_cast` 必抛），
> 旋转 key 去重那条**必须移到 PR5**（它读 PR5 才有的 `maxLevel`），
> `Ciphertext.cpp` 横跨 **2/4/5** 而不是 4/7。
>
> **2026-09-11 二次更新**：上游 `main` 前进到 `fa97286`（三个 commit，`786c760..fa97286`，
> 只动了 3 个文件 11 行）。已 merge 进 `bootstrap-dev`，**无冲突，我方代码不需要任何修改**，
> `patch/` 已按新基线重切（重建校验与前向引用校验都仍然通过，块划分不变）。
> 两处上游修复都落在我们也改的文件上，而且**方向和我们一致**：
>
> * `rotate_hoisted` 非融合路径改用 `copyMetadata(*this)`，此前 `slots` 一直是 0。
>   这不只是元数据不全：`GetRotationKey(index, keyID, slots, actual_index)` 在精确 index 不存在时
>   会走「模 slot 兼容」回退，循环写作 `for (i = 1; i < N/2/slots; ++i)`——**slots 为 0 就是整数除零**。
>   我们的 level 截断密钥正是在这条路径上调 `ensureLevel`，所以这条修复对我们是纯收益。
> * `LimbPartition::multPt` 改用 `limb.at(limbsize - 1)` 而不是 `limb.back()`：
>   `dropToLevel` 不缩物理存储，所以 `back()` 取到的可能是当前 level 之外的那根 limb。
>   **light plaintext 恰好是最容易踩到的场景**——同一个池化多项式被反复在不同 level 上展开。
>   这和我们在 `LTdotProductPtBatch` / `multMonomial` 加的 limb 检查是同一类问题。
>   顺带：`LimbPartition.cu:2438` 还留着一个 `STREAM(limb.back())`，是流等待不是数据操作，
>   但同一类隐患，可以提给上游。

---

## 一、范围

| | 文件数 | 行数 |
|---|---:|---:|
| **进 PR（引擎）** | 48 | +3488 / -52 |
| 不进 PR（THOR 移植、过程记录） | 70 | +11749 / -0 |
| 分支合计 | 118 | +15237 / -52 |

进 PR 的按目录：

| 目录 | 文件 | 行数 |
|---|---:|---:|
| `src/`（CUDA 内核与 CKKS 核心） | 22 | +1121 / -39 |
| `api/`（OpenFHE 兼容层） | 8 | +697 / -12 |
| `python/`（pybind11 绑定、包装、引擎侧 pytest） | 12 | +926 |
| `examples/key-truncation/` | 3 | +361 |
| `docs/` | 2 | +247 |
| `tools/memory_model.py` | 1 | +70 |
| `CMakeLists.txt`、`.gitignore` | 2 | +66 / -1 |

不进 PR 的：`python/thorfhe/`（32 个模块）、`python/tests/test_stage6..17_thor_*.py`、
`python/tests/test_gpu_resource_fixes.py`（**它 import thorfhe**，见第三节）、
`report/`、`bugs/`、`docs/thor_port.md`、`docs/thor-engine-requirements.md`、`docs/benchmark.md`。

---

## 二、建议拆成 8 个 PR，不要一次性提

3488 行一次提审基本审不动，而且里面四类东西的风险完全不同：**纯 bug 修复**（应当优先合，
上游任何用户都受影响）、**新 API**（要讨论命名和语义）、**可选特性**（要讨论默认值）、
**构建脚本**（无风险）。混在一起会让最该合的那部分被最需要讨论的那部分拖住。

顺序即依赖顺序。1、2 可以立刻提，互不依赖。

| # | 主题 | 大致行数 | 依赖 | 风险 |
|---|---|---:|---|---|
| 1 | 构建：GPU 架构自动探测 | 66 | 无 | 无 |
| 2 | 七个正确性修复（不含新 API） | 约 110 | 无 | 低，全是收紧检查 |
| 3 | CKKS 槽级原语 | 约 200 | 无 | 低，纯新增 |
| 4 | 惰性重线性化（degree-2 密文） | 约 250 | 无 | 中，动了 `Ciphertext` |
| 5 | level 截断密钥存储 | 约 900 | 2 | 中，默认开启需讨论 |
| 6 | light plaintext（紧凑明文） | 约 450 | 无 | 低，纯新增 |
| 7 | 显存池：回收、排空、可观测 | 约 300 | 无 | 中，动了分配器 |
| 8 | Python 绑定与 pytest | 约 926 | 3–7 | 低 |

**有六个文件横跨多个 PR**，拆分要按 hunk cherry-pick，不能按文件：`api/CryptoContext.{hpp,cpp}`、
`api/Definitions.hpp`、`src/CKKS/Context.{cu,cuh}`、`src/CKKS/KeySwitchingKey.cuh`、
`src/CKKS/LimbPartition.{cu,cuh}`、`src/CKKS/RNSPoly.{cpp,cuh}`、`src/CKKS/Ciphertext.cpp`、
`src/CKKS/openfhe-interface/RawCiphertext.cu`。最极端的是 `api/CryptoContext.cpp` 里一个 374 行的
hunk，横跨 PR 3/4/5/6/7。已按行区间切好，见 `../patch/_tools/manifest.py`。

### PR 1 — 构建：GPU 架构自动探测

`CMakeLists.txt` 原本把 `FIDESLIB_ARCH` 硬编码成 `80..120`。结果：V100（sm_70）编不出来，
而 CUDA 13 已经删掉 sm_70、CUDA < 12.8 又不认识 sm_100/120，硬编码在两头都会炸。
改成从 `nvcc --version` 推出支持区间、从 `nvidia-smi` 读实际在场的卡，取交集；
`-DFIDESLIB_ARCH=70-real` 仍可强制指定。落在区间外的卡给 warning 而不是失败。

文件：`CMakeLists.txt`、`.gitignore`。

### PR 2 — 七个正确性修复

都是「原来会静默出错，现在要么正确要么报错」，没有新 API。**我认为这是最该优先合的一个 PR。**

1. **`MAXP` 溢出没有任何检查**（`src/ConstantsGPU.cu`）。所有常量表都是 `[MAXP]`，而 `MAXP`
   同时是内核索引扁平指针表时的**步长**（`i * MAXP + primeid`），特殊素数写在 `primes[L + i]`，
   所以需要 `L + K <= MAXP`。超了不报错，而是越过 `primes` 写进 `prime_better_barret_mu`，
   症状取决于越界写到哪儿。在 GV100 / N=2^16 / dnum=3 上观察到的顺序是：先是某个算子静默产出
   NaN，然后 illegal memory access，最后 `cudaMemcpy 'invalid argument'`——**同一个 bug 被当成三个
   bug 报了三次**。现在带着三个数字抛异常，并说明哪几个参数能改变它们。
2. **`EvalCoeffsToSlots` 的明文/密文 level 差一**（`src/CKKS/CoeffsToSlots.cu`）。对角线由
   `EvalBootstrapSetup` 编码在它自己选的 level，而 `ModRaise` 把密文抬到 `cc.L`
   （只有 FLEXIBLEAUTOEXT 会减一）。在不做隐式 level 调整的 scaling technique 下两者差一个 limb，
   批量乘法内核按密文尺寸发射、用同一个界索引明文，于是**读出明文末尾之后一个 limb**。
   修法是每个 LT step 之前把密文降到该 step 对角线的 level——即 OpenFHE 的
   `AdjustLevelsAndDepth` 隐式在做的事。**这个修复之前，GPU 上从未成功完成过一次 bootstrap。**
3. **`LTdotProductPtBatch` 没有边界检查**（`src/CKKS/LimbPartitionBatch.cu`）。发射尺寸取自
   `out[0]` 的 level，内核只检查 `pt_partition` 非空就去访问 `pt_partition[blockIdx.y]`。
   一个存在但 limb 数不足的分区会得到越界指针，illegal access 在下一次同步时才浮现。
   现在提前检查并报出「要读 N limb、实际只有 M」。**注意**：`pt` 里的 nullptr 是合法的
   （`DotProductPtInternal` 会主动压入 nullptr，内核有两处 `!= nullptr` 保护），检查必须跳过它们。
4. **`multMonomial` 没有检查辅助多项式的 limb 数**（`src/CKKS/Ciphertext.cpp`）。
   `RNSPoly::grow` 在池化多项式已达目标 level 时提前返回，所以 `g.limb[i]` 越界是可能的，
   后果同上：垃圾设备指针，illegal access 在几次调用之后的 `GPUfree` 里才炸——这正是它最初
   在 GV100 上被报出来的位置。就地检查并说清是哪两个数字。
5. **`KeySwitchingKey` 持有悬垂引用**（`src/CKKS/KeySwitchingKey.cuh`）。原来按引用持有
   `Context`，而 `LoadContext` 会把局部 `Context` move 进 `std::any`，函数返回后引用即悬垂。
   **改成 `std::weak_ptr<ContextData>` 加一个 `context()` 访问器。**
   （中间版本是按值持有 `shared_ptr`，那是错的：key 活在 `ContextData::precom.keys` 里面，
   持有强引用就成了环，`ContextData` 永远归零不了，每个建了又销毁的 context 都把 eval key、
   rotation key、bootstrap 明文和辅助缓冲留到进程结束。当时的注释把它写成「deliberate」，
   是给一个漏内存的设计找说法；2026-09-11 的 review 指出了这一点，已改。）
   weak_ptr 在 key 活着时不可能过期——key 就是它的成员——`context()` 里 assert 了这一点。
   调用方在函数开头取一次 `Context cc = context();`，函数体不用改；
   `Context.cu::AddSecretSwitchingKey` 两处 `ksk.cc->param` 改成 `ksk.context()->param`。
   同一 PR 还修了析构时误清全局 key 表、`SetDevices` 只接受右值两处。
6. **`EvalSub(double scalar, const Ciphertext& ct)` 符号反了**（`api/CryptoContext.cpp`）。
   原实现是 `multScalar(-1)` → `addScalar(scalar)` → `multScalar(-1)`，算出来是 `ct - scalar`；
   而这个重载的语义是 `scalar - ct`。去掉最后那次取反即可。**这是切分代码时才发现的**，
   不在原来的清单里。
7. **`EvalMult(ct, pt)` / `EvalMultInPlace(ct, pt)` 的 CPU 回退路径必抛**（`api/CryptoContext.cpp`）。
   `std::any_cast<const lbcrypto::ConstPlaintext&>(pt->cpu)`，而 `pt->cpu` 里装的是 `Plaintext`。
   `any_cast` 要求类型精确匹配，所以这条路径一走就 `bad_any_cast`。改成 `const lbcrypto::Plaintext&`。
   同样是切分时发现的。

> **旋转 key 去重那条修复原本列在这里，已移到 PR 5。** 它的判断读
> `KeySwitchingKey::maxLevel`——PR 5 才引入的成员。也就是说**那个 bug 只有在密钥可以被截断之后
> 才存在**，放进「纯修复」PR 里既编不过也讲不通。

### PR 3 — CKKS 槽级原语

一组小而独立的新算子，都是 OpenFHE 有语义、GPU 后端此前没有的：

| API | 语义 | 代价 |
|---|---|---|
| `EvalConjugate` / `EvalConjugateKeyGen` | 逐槽复共轭（自同构 index `2N-1`） | 一次 key switch，不耗 level |
| `EvalMultByI` | 逐槽乘 i（单项式 `X^(N/2)`） | 免费：无 key switch、无 level、scale 不变 |
| `EvalMultByInteger` | 乘小整数 | 不耗 level、不改 scale |
| `EvalLevelReduce` | 丢 `levels` 个 RNS limb 而**不** rescale | 免费 |
| `GetRemainingLevels` / `GetConsumedLevels` | 剩余 / 已耗 level | 查询 |
| `CCParams::SetCKKSDataType` | 透传 OpenFHE 的 REAL / COMPLEX 槽编码 | 参数 |

`EvalLevelReduce` 与 `GetConsumedLevels` 成对使用：后者返回的正是
`MakeCKKSPackedPlaintext` 期望的 `level` 参数，手工管理 scale 时需要它来对齐明文与密文。

文件：`api/CryptoContext.{hpp,cpp}`、`api/CCParams.{hpp,cpp}`、`api/Definitions.hpp`。

### PR 4 — 惰性重线性化

`Ciphertext` 增加可选的第三分量 `c2`（`std::unique_ptr<RNSPoly>`）。
`multNoRelin` / `squareNoRelin` 产出 degree-2 密文并把 `c1*d1` 留在 `c2`；
线性操作（add、sub、乘明文、乘标量、rescale、dropToLevel、copy）对 `c2` 同样生效；
`relinearize()` 用 eval key 把 `c2` 切回 `(c0, c1)`，degree-1 输入是 no-op。

意义在于累加式内积：`sum_k a_k * b_k` 原来每一项一次 key switch，现在整个和只要一次。
key switch 是这类电路的主要开销。

文件：`src/CKKS/Ciphertext.{cpp,cuh}`，`api/CryptoContext.{hpp,cpp}` 的
`EvalMultNoRelin` / `EvalSquareNoRelin` / `EvalRelinearize`。

### PR 5 — level 截断密钥存储

一把旋转 key 的 DECOMP/DIGIT limb 数按最高 level 分配，但**大多数 key 从来不在最高 level 被用**。
按每把 key 实际会被用到的最高 level 只驻留必要的 limb，其余不上设备。

* `KeySwitchingKey::maxLevel`（-1 = 完整，即原行为）、`ensureLevel(level)`、`coversLevel`、
  `rebuildAtLevel`、`deviceBytes`、`describe`；
* `RNSPoly` / `LimbPartition` 的 `generateDecompAndDigit(iskey, maxLevel)`、`resetDecompAndDigit`、
  `decompDigitLevelCovered` / `decompDigitMaxLevel` / `decompDigitDeviceBytes`；
* `CryptoContextImpl::SetRotationKeyLevels(map<index, maxRemainingLevels>)`，须在 `LoadContext` 前调用；
* `GetBootstrapKeyLevelPlan(cc, slots, GPUcc)`：自动算出 bootstrap 自己那批 key 的 level 计划——
  StC 的 key 只在 `EvalMod` 之后使用，因此上界是 `L - GetBootstrapDepth + levelBudget[1]`；
* 可观测：`GetKeyDeviceBytes()`、`GetGrownKeyCount()`、`ContextData::printKeyMemoryReport`；
* 开关：`truncate_keys`（默认 true）、`key_level_margin`（默认 1）、`allow_key_grow`（默认 false），
  三者都可用环境变量覆盖（`FIDESLIB_KEY_TRUNCATION` / `FIDESLIB_KEY_LEVEL_MARGIN` /
  `FIDESLIB_KEY_GROW`）。

**默认值需要讨论。** 现在 `allow_key_grow` 默认 false，即 level 计划算错时**抛异常而不是悄悄
重建 key**：一个错的计划是调用方的 bug，静默重载会同时隐藏成本和错误。但对上游既有用户来说，
`truncate_keys` 默认 true 是行为变更；保守起见可以在上游改成默认 false。

**含一个只有截断存在时才存在的 bug 的修复**：同一个旋转 index 会被注册两次——一次来自
`SetRotationKeyLevels`（调用方电路旋转的 level），一次来自 bootstrap 预计算（StC / CtS 旋转的
level），两组 index 天然重叠。`std::map::emplace` 保留先到的、静默丢弃后到的，于是留下的可能是
截断过头的那把，随后在完全合法的使用点抛错。改成保留覆盖更多的那把（完整 key 胜过任何截断 key）。
实现上要 erase + emplace 而不是赋值——`KeySwitchingKey` 含 const 成员，不可赋值。
`RawCiphertext.cu` 里 caller 一侧有对应的两处。

配套：`examples/key-truncation/`（两个可执行：一个测量节省量，一个复现 grow 路径）、
`docs/level_truncated_keys.md`、`tools/memory_model.py`（按 OpenFHE 的 BSGS 参数化估算 key 驻留量）。

### PR 6 — light plaintext

一个编码后的明文占 `(L+1)` 个 RNS tower；N=2^16、L=30 时约 16 MiB。但编码结果本质上只是
`round(Delta * IFFT(message))` 的 **N 个中心化整数系数**——0.5 MiB，且**与 level 无关**。
先存系数、用到时再在设备上展开成 tower，权重量大的应用能省两个数量级的内存和 PCIe 流量。

* `LightPlaintext` / `LightPlaintextImpl`（`api/LightPlaintext.{hpp,cpp}`），带 uid、可存盘；
* `MakeLightPlaintext(value, slots, levelHint)`，复数与实数两个重载；
* `ExpandLightPlaintext(lp, level)`；GPU 上直接由系数在设备侧生成 tower，
  **只有 N * 8 字节过 PCIe**（`expandCentredCoeffs_` 内核 +
  `LimbPartition::loadCentredCoefficients` + `RNSPoly::loadCentredCoefficients` +
  `Plaintext::loadLight`）；
* `EvalMult(ct, lp)` / `EvalAdd(ct, lp)`：按密文当前 level 展开，走 FIFO 缓存
  （`light_plaintext_cache_capacity`，默认 64，0 关闭）、`ClearLightPlaintextCache()`。

**前提条件要写清楚**：只在 Delta 不随 level 变化的 scaling technique（FIXEDMANUAL / FIXEDAUTO）下
成立。FLEXIBLE* 下系数只在 `levelHint` 那一级有效，在别处展开会抛异常——这就是 `levelHint` 存在的
理由。

配套：`docs/light_plaintext.md`。

### PR 7 — 显存池：回收、排空、可观测

三段式，缺一段另外两段都白做。

1. **池子从不归还**（`src/CudaUtils.cu`）。池按**精确分配尺寸**分类，一个涨起来的 size class
   无法把内存借给没涨起来的。观察到的死法是：还剩 16 MiB 空闲，一个 512 KiB 的 slab 请求失败，
   那 16 MiB 全在别的 size class 的空闲表里。新增 `ReclaimFreeSlabs`：把所有 block 都空闲的整块
   slab 还给 driver。**只在分配即将失败时触发，快路径不受影响。**
2. **析构密文不释放显存**（`src/CKKS/Context.cu`、`Ciphertext.cpp`）。`~Ciphertext` 把多项式
   停进 context 的辅助池给下一个复用，而**全项目没有任何地方排空那个池**，于是它稳定在
   「同时活着的密文数」的历史最高水位并一直占着。这正是上一条「回收跑了但一块都没找到」的原因：
   所有块都还被辅助池攥着，从空闲表的角度看它们全都在用。新增 `TrimAuxiliaryPolys(keep = 0)`
   与 `GetAuxiliaryPolyCount()`。建议在阶段边界调用——中途排空是负收益，那些多项式马上又要被取回。
3. **无法观测**（`src/CudaUtils.{cu,cuh}`）。
   `PoolStats { pooled, in_use, driver_free, driver_total }` 与 `GetPoolStats(id)`，
   经 `CryptoContextImpl::GetDeviceMemory()` 暴露。`in_use` 是推出来而非数出来的
   （池拥有的每块要么发出去了要么在空闲表里），因此精确且不需要在分配快路径上加任何记账。
   有了这四个数才能区分「池在囤」和「密钥明文本来就占满了」——只看 OOM 分不出来。

同一 PR 还包含 slab 尺寸自适应减半：固定 1 GiB 的 slab 在余量不足时必然失败，改成申请不到就折半重试。

### PR 8 — Python 绑定与测试

`python/src/bindings.cpp`（pybind11）+ `python/pyfideslib/__init__.py`（薄封装：上下文构造、
key 生成、上面各 PR 的算子）+ `python/CMakeLists.txt` + `python/README.md`。

pytest 五个文件，按能力分层，无 GPU 时自动 skip：

| 文件 | 覆盖 |
|---|---|
| `test_stage1_io.py` | 编码 / 加密 / 解密往返 |
| `test_stage2_linear.py` | 加减、乘明文、旋转、共轭、乘 i、level reduce |
| `test_stage3_lazy_relin.py` | degree-2 密文与单次重线性化（PR 4） |
| `test_stage4_bootstrap.py` | 满槽复数 bootstrap 与输出 level 契约 |
| `test_stage5_light_plaintext.py` | light plaintext 的编码、展开、缓存、存盘（PR 6） |

`conftest.py` 提供 fixture 与 GPU 检测。

---

## 三、提交前必须做的：THOR 信息隔离

引擎代码里目前**残留 55 处 THOR 字样，分布在 20 个文件**。它们都是注释或文档，不影响编译，
但会把一个外部工作负载的名字带进上游。逐处处理如下。

### 3.1 整份排除

| 文件 | 处理 |
|---|---|
| `docs/thor_port.md` | 不进 PR |
| `docs/thor-engine-requirements.md` | 不进 PR |
| `docs/benchmark.md` | 不进 PR（写的是 `thorfhe.bench` 的用法） |
| `python/tests/test_gpu_resource_fixes.py` | **不进 PR**——它 `import thorfhe`，是 THOR 电路的测试，不是引擎测试。文件名有误导性 |
| `report/`、`bugs/` | 不进 PR |

注意 `api/CryptoContext.hpp:169` 与 `python/pyfideslib/__init__.py:4` 里有指向
`docs/thor-engine-requirements.md` 的**交叉引用**，那份文档不进 PR，这两处引用必须一并改掉，
否则上游会出现指向不存在文件的链接。

### 3.2 逐处改写

| 文件:行 | 现状 | 建议 |
|---|---|---|
| `api/CryptoContext.hpp:169` | `// ---- THOR-style primitives (see docs/thor-engine-requirements.md) ----` | `// ---- Slot-level primitives ----` |
| `api/CryptoContext.cpp:2051` | `// ---- THOR-style primitives ----` | 同上 |
| `api/CryptoContext.hpp:176` | `(THOR's DeltaCiphertext trick)` | 删括号，语义本身已说清 |
| `api/CryptoContext.hpp:178` | `(THOR's level_down)` | 删 |
| `api/CryptoContext.hpp:187` | `THOR's encode_to_light_plaintext` | 删 |
| `api/CryptoContext.hpp:242` | `(THOR's create_fixed_rotation_key(sk, delta, level))` | 删 |
| `api/LightPlaintext.hpp:23` | `That is what THOR's weights need. BERT-base has ~220k ...` | 改成「一个带大量固定权重明文的工作负载，例如 BERT-base 规模的加密推理约需 22 万个编码明文」 |
| `api/LightPlaintext.hpp:46` | `THOR encodes weights at a fixed level` | 改成「调用方通常在固定 level 编码权重」 |
| `api/LightPlaintext.cpp:22` | `THOR reads hundreds of thousands of these` | 改成「调用方可能读取数十万个」 |
| `src/CKKS/Ciphertext.cuh:60` | `(lazy relinearisation, THOR style)` | `(lazy relinearisation)` |
| `src/CudaUtils.cu:366` | `running a full THOR layer` | `running a full transformer layer` |
| `src/CudaUtils.cu:427` | `every stage boundary of a THOR layer` | `every stage boundary of a large circuit` |
| `python/src/bindings.cpp:151,169` | 两处 docstring | 删 THOR 字样 |
| `python/pyfideslib/__init__.py:4,28,117,209,213` | 五处 | `THOR-style` 改为 `slot-level`；交叉引用改成 `docs/light_plaintext.md` |
| `examples/key-truncation/src/key_truncation.cpp:14,15` | 用 THOR 举例说明省了多少 | 改成中性描述：「按约 250 把固定 level 的旋转 key 计算」 |
| `docs/level_truncated_keys.md:45,85,89` | 三处 | 改中性；`python/thorfhe` 的交叉引用删掉 |
| `docs/light_plaintext.md:5,20,56,74,103,115` | 六处 | 同上。**第 20 行引用了 `THOR/src/thor/he.py`（外部私有路径），必须删** |

### 3.3 一条检查命令

清理完后应当返回空：

```
grep -rn "THOR" src/ api/ python/src/ python/pyfideslib/ examples/key-truncation/ docs/
```

---

## 四、PR 正文（英文，可直接粘贴）

### PR 1

> **Build: derive CUDA architectures from the toolkit and the installed GPUs**
>
> `FIDESLIB_ARCH` was hard-coded to `80-real;86-real;89-real;90-real;90-virtual;100-real;120-real`.
> That fails at both ends of the supported range: a V100 (sm_70) is not in the list, CUDA 13 no longer
> accepts sm_70, and toolkits older than 12.8 do not know sm_100 or sm_120.
>
> The architecture list is now the intersection of what `nvcc --version` supports and what
> `nvidia-smi` reports as present, with a warning rather than an error for a GPU outside the toolkit's
> range. `-DFIDESLIB_ARCH=...` still overrides it entirely.

### PR 2

> **Fix seven silent-corruption paths**
>
> Each of these previously produced wrong results or an illegal memory access at a point far from the
> cause. No new API.
>
> 1. `MAXP` overflow was unchecked. Every constant table is `[MAXP]`, and `MAXP` is simultaneously the
>    stride the kernels index the flattened pointer tables with, so the scheme requires
>    `L + K <= MAXP`. Exceeding it writes past `primes` into the following table. On a GV100 at
>    N=2^16, dnum=3 this surfaced first as a silent NaN, then as an illegal memory access, then as
>    `cudaMemcpy 'invalid argument'` - one bug reported as three. It now throws with the three numbers
>    and says which parameters move them.
> 2. `EvalCoeffsToSlots` read one limb past the end of its plaintexts. The bootstrap diagonals are
>    encoded by `EvalBootstrapSetup` at a level of its own choosing, while `ModRaise` grows the
>    ciphertext to `cc.L`; under a scaling technique that does not adjust levels implicitly the two
>    disagree by one limb, and the batched product sizes its launch from the ciphertext and indexes
>    the plaintext with the same bound. The ciphertext is now dropped to the diagonals of the step
>    about to run, per step, which is what `AdjustLevelsAndDepth` does implicitly on the CPU path.
> 3. `LTdotProductPtBatch` had no bounds check. A partition that is present but holds fewer limbs than
>    the launch implies yields an out-of-range pointer whose illegal access appears at the next
>    synchronisation. It is checked up front now, with both limb counts in the message. A null `pt`
>    entry is legitimate and is skipped: `DotProductPtInternal` pushes nullptr deliberately and the
>    kernel guards for it.
> 4. `multMonomial` did not check its auxiliary polynomial's limb count. `RNSPoly::grow` returns
>    early when the pooled polynomial is already at or above the target level, so an indexing overrun
>    is possible, with the same consequence as above: a garbage device pointer whose illegal access
>    surfaces several calls later, in a `GPUfree`, which is where it was first reported from.
> 5. `KeySwitchingKey` held a dangling `Context` reference: `LoadContext` moves its local context into
>    a `std::any`, so the reference dies when `LoadContext` returns. It is a
>    `std::weak_ptr<ContextData>` now, reached through a `context()` accessor. Holding a `Context` by
>    value would fix the dangle too, but the keys live inside `ContextData::precom.keys`, so a strong
>    reference closes a cycle and every context dropped through the API keeps all of its device
>    memory. The weak_ptr cannot expire while the key is alive, for exactly that reason.
>    Two smaller fixes ride along: the destructor wiped the global key map, and `SetDevices` only
>    accepted an rvalue.
> 6. `EvalSub(double scalar, const Ciphertext& ct)` had its sign inverted. It computed
>    `multScalar(-1)`, `addScalar(scalar)`, `multScalar(-1)`, which is `ct - scalar`, while the
>    overload means `scalar - ct`. The trailing negation is removed.
> 7. The CPU fallback in `EvalMult(ct, pt)` and `EvalMultInPlace(ct, pt)` always threw:
>    `std::any_cast<const lbcrypto::ConstPlaintext&>` on an `any` holding a `Plaintext`. `any_cast`
>    requires an exact type match, so that path raised `bad_any_cast` every time it was taken.

### PR 3

> **CKKS slot-level primitives: conjugation, multiply-by-i, integer scaling, level reduction**
>
> Operations OpenFHE defines but the GPU backend did not expose. `EvalConjugate` (with
> `EvalConjugateKeyGen`) is one key switch and costs no level. `EvalMultByI` is the monomial
> `X^(N/2)`: no key switch, no level, no change of scale. `EvalMultByInteger` scales by a small
> integer without consuming a level. `EvalLevelReduce` drops RNS limbs without rescaling.
> `GetRemainingLevels` and `GetConsumedLevels` report a ciphertext's position in the modulus chain;
> the latter returns exactly the `level` argument `MakeCKKSPackedPlaintext` expects, which is what
> aligning a plaintext to a ciphertext requires when scales are managed manually.
>
> `CCParams::SetCKKSDataType` forwards OpenFHE's REAL/COMPLEX slot encoding choice.

### PR 4

> **Lazy relinearisation: degree-2 ciphertexts**
>
> `Ciphertext` gains an optional third component `c2`. `multNoRelin` and `squareNoRelin` leave the
> `c1*d1` term there instead of key-switching it away; linear operations (add, sub, plaintext and
> scalar multiplication, rescale, level reduction, copy) act on it as well, and `relinearize()`
> switches it back onto `(c0, c1)` with the evaluation key.
>
> The point is accumulating inner products: `sum_k a_k * b_k` needed one key switch per term and now
> needs one for the whole sum. Key switching dominates the cost of such circuits.

### PR 5

> **Level-truncated key storage**
>
> A rotation key's DECOMP/DIGIT limbs are allocated for the top level, but most keys are never applied
> there. `SetRotationKeyLevels` takes a map from rotation index to the highest level that key will
> ever be used at, and only the limbs required for it are made resident.
>
> `GetBootstrapKeyLevelPlan` computes the same plan for the bootstrap's own keys: the slots-to-coeffs
> keys are used only after `EvalMod`, so their bound is `L - GetBootstrapDepth + levelBudget[1]`.
>
> Registering the same rotation index twice used to keep the wrong key: `std::map::emplace` keeps the
> first and silently drops the second, so a key truncated for the caller's circuit could survive and
> then be used by the bootstrap above the level it was cut to. The key that covers more now wins.
> This fix lives here rather than with the other bug fixes because it reads `KeySwitchingKey::maxLevel`
> - the bug only exists once keys can be truncated at all.
>
> A key needed above its plan is a caller bug, so `ensureLevel` throws by default, naming the key and
> both levels; `allow_key_grow` (or `FIDESLIB_KEY_GROW=1`) rebuilds it instead. `GetKeyDeviceBytes`
> and `GetGrownKeyCount` make the plan checkable: a correct plan grows zero keys.
>
> `truncate_keys` defaults to true here. Reviewers may prefer false upstream, since it is a behaviour
> change for existing users; the switch and its environment variable are in place either way.
>
> Includes `examples/key-truncation` (one binary measuring the saving, one reproducing the grow path),
> `docs/level_truncated_keys.md`, and `tools/memory_model.py`, which estimates key residency from
> OpenFHE's BSGS parametrisation.

### PR 6

> **Light plaintexts: compact, level-agnostic encoded plaintexts**
>
> An encoded plaintext occupies `(L+1)` RNS towers - about 16 MiB at N=2^16, L=30 - but the encoding
> is really the N centred integer coefficients of `round(Delta * IFFT(message))`, which is 0.5 MiB and
> does not depend on the level. `MakeLightPlaintext` stores those; `ExpandLightPlaintext` builds the
> towers at whatever level is needed, on the device, so only `N * 8` bytes cross PCIe.
>
> `EvalMult(ct, lp)` and `EvalAdd(ct, lp)` expand at the ciphertext's level through a FIFO cache
> (`light_plaintext_cache_capacity`, default 64, 0 disables it).
>
> This is only valid when Delta does not depend on the level, i.e. FIXEDMANUAL or FIXEDAUTO. Under
> FLEXIBLE* the coefficients are valid only at `levelHint`, and expanding elsewhere throws rather than
> returning a quietly wrong plaintext.

### PR 7

> **Device memory: return free slabs, drain the auxiliary pool, and report both**
>
> Three changes that only work together.
>
> The pool is keyed by exact allocation size and never returned anything, so a size class that had
> grown could not lend to one that had not: a 512 KiB request failing with 16 MiB free, all of it in
> other classes' free lists. `ReclaimFreeSlabs` returns slabs whose blocks are all free, and runs only
> when an allocation is about to fail, so the fast path is untouched. Slab acquisition also halves and
> retries rather than failing outright on a fixed 1 GiB request.
>
> That alone reclaims nothing, because destroying a ciphertext does not free its polynomials - it
> parks them in the context's auxiliary pool for the next one to reuse, and nothing ever drained that
> pool, so it settled at the high-water mark of live ciphertexts and held it. From the free lists'
> point of view every block was still in use. `TrimAuxiliaryPolys` drains it; a stage boundary is the
> right place to call it, since mid-stage those polynomials are about to be reused.
>
> `GetPoolStats` / `GetDeviceMemory` report `{pooled, in_use, driver_free, driver_total}`. `in_use` is
> derived by subtracting the free lists from the slabs rather than counted, so it is exact and adds no
> bookkeeping to the allocation path. Without it, an out-of-memory failure cannot distinguish a pool
> that is hoarding from keys and plaintexts that genuinely fill the card.

### PR 8

> **Python bindings and a layered test suite**
>
> pybind11 bindings for the context, key generation and the operations added by the preceding PRs,
> plus a thin `pyfideslib` wrapper over the usual setup, and five pytest files that skip themselves
> when no GPU is present: encode/encrypt/decrypt round trips, the linear operations, degree-2
> ciphertexts and single relinearisation, full-slot complex bootstrap and its output level contract,
> and light plaintext encoding, expansion, caching and persistence.

---

## 五、THOR 侧（不进 PR，仅供对照）

上面任何一节都不含 THOR 的算法细节；这一节说明被排除的是什么，以及每项引擎特性的来历——
写在这里，就不必写进 PR。

分支上 70 个文件、约 11749 行属于 THOR 移植：`python/thorfhe/` 的 32 个模块（几何、编码、
attention、softmax、layernorm、GELU、feed-forward、pooler、benchmark CLI、精确算术参考引擎）、
`python/tests/test_stage6..17_thor_*.py` 与 `test_gpu_resource_fixes.py`、`report/`、`bugs/`、
`docs/thor_port.md`、`docs/thor-engine-requirements.md`、`docs/benchmark.md`。

| 引擎特性 | 由什么驱动 |
|---|---|
| 槽级原语（PR 3） | 复数打包的注意力需要逐槽共轭与乘 i；手工 scale 管理需要 level reduce |
| 惰性重线性化（PR 4） | 密文-密文矩阵乘的累加内积 |
| level 截断密钥（PR 5） | 单卡 32 GiB 放不下约 250 把满 level 旋转 key |
| light plaintext（PR 6） | BERT-base 规模约 22 万个权重明文 |
| 显存池三件套（PR 7） | 单层 transformer 的工作集与密钥、明文争同一张卡 |
| `MAXP` 检查（PR 2.1） | 深电路把 `L + K` 顶过了 64 |
| CtS 对齐（PR 2.2） | 手工 scale 管理与 bootstrap 预计算之间的接缝 |

---

## 六、验证状态（如实）

* **开发机没有 GPU。** 所有 C++ 的编译与 GPU 运行都在一台远程 GV100（32 GiB，sm_70，CUDA 12.9）上，
  由对侧执行；本地只能保证 Python 侧。
* **已在 GPU 上确认可用**：light plaintext 全路径；level 截断密钥（含 `0 grown at runtime`）；
  bootstrap（CtS 对齐修复之后首次成功完成，误差 1.74e-5）；PR 2 的五个修复；
  一个 transformer 层的前若干阶段（relRMSE 1.03e-8）。
* **尚未编译过**：`GetPoolStats` / `GetDeviceMemory`（2026-09-10 新增）。**提 PR 前必须先编译。**
* **Python 套件**：91 passed / 40 skipped（skip 的都是需要 GPU 的用例）。
* **上游回归：本分支没有跑过上游自带的 C++ 测试。**（`fa97286` 已 merge，同样没编译过。） PR 2、4、7 动了公共路径
  （`Ciphertext`、`CoeffsToSlots`、分配器），提交前必须跑一遍上游 test suite。
  **这是目前最大的空白，也是提 PR 前唯一的硬阻塞。**

---

## 七、完整文件清单（引擎侧 48 个）

| 文件 | +/- | PR |
|---|---:|---|
| `.gitignore` | +4 / -0 | 1 |
| `CMakeLists.txt` | +62 / -1 | 1 |
| `api/CCParams.cpp` | +7 / -0 | 3 |
| `api/CCParams.hpp` | +2 / -0 | 3 |
| `api/CryptoContext.cpp` | +414 / -12 | 2,3,4,5,6,7 |
| `api/CryptoContext.hpp` | +107 / -0 | 3,4,5,6,7 |
| `api/Definitions.hpp` | +12 / -0 | 3,6 |
| `api/LightPlaintext.cpp` | +83 / -0 | 6 |
| `api/LightPlaintext.hpp` | +71 / -0 | 6 |
| `api/fideslib.hpp` | +1 / -0 | 6 |
| `docs/level_truncated_keys.md` | +128 / -0 | 5 |
| `docs/light_plaintext.md` | +119 / -0 | 6 |
| `examples/key-truncation/CMakeLists.txt` | +44 / -0 | 5 |
| `examples/key-truncation/src/key_grow_repro.cpp` | +168 / -0 | 5 |
| `examples/key-truncation/src/key_truncation.cpp` | +149 / -0 | 5 |
| `python/CMakeLists.txt` | +22 / -0 | 8 |
| `python/README.md` | +22 / -0 | 8 |
| `python/pyfideslib/__init__.py` | +268 / -0 | 8 |
| `python/src/bindings.cpp` | +226 / -0 | 8 |
| `python/tests/conftest.py` | +43 / -0 | 8 |
| `python/tests/test_stage1_io.py` | +25 / -0 | 8 |
| `python/tests/test_stage2_linear.py` | +104 / -0 | 8 |
| `python/tests/test_stage3_lazy_relin.py` | +45 / -0 | 8 |
| `python/tests/test_stage4_bootstrap.py` | +44 / -0 | 8 |
| `python/tests/test_stage5_light_plaintext.py` | +127 / -0 | 8 |
| `src/CKKS/Ciphertext.cpp` | +154 / -9 | 2,4,5 |
| `src/CKKS/Ciphertext.cuh` | +20 / -0 | 4 |
| `src/CKKS/CoeffsToSlots.cu` | +25 / -1 | 2 |
| `src/CKKS/Context.cu` | +88 / -1 | 2,5,7 |
| `src/CKKS/Context.cuh` | +20 / -0 | 5,7 |
| `src/CKKS/ElemenwiseBatchKernels.cu` | +18 / -0 | 6 |
| `src/CKKS/ElemenwiseBatchKernels.cuh` | +7 / -0 | 6 |
| `src/CKKS/KeySwitchingKey.cu` | +81 / -10 | 2,5 |
| `src/CKKS/KeySwitchingKey.cuh` | +51 / -1 | 2,5 |
| `src/CKKS/LimbPartition.cu` | +137 / -4 | 5,6 |
| `src/CKKS/LimbPartition.cuh` | +21 / -2 | 5,6 |
| `src/CKKS/LimbPartitionBatch.cu` | +25 / -0 | 2 |
| `src/CKKS/LinearTransform.cu` | +1 / -0 | 5 |
| `src/CKKS/Plaintext.cu` | +10 / -0 | 6 |
| `src/CKKS/Plaintext.cuh` | +14 / -0 | 6 |
| `src/CKKS/RNSPoly.cpp` | +62 / -2 | 5,6 |
| `src/CKKS/RNSPoly.cuh` | +18 / -1 | 5,6 |
| `src/CKKS/openfhe-interface/RawCiphertext.cu` | +182 / -5 | 2,5 |
| `src/CKKS/openfhe-interface/RawCiphertext.cuh` | +15 / -0 | 5 |
| `src/ConstantsGPU.cu` | +20 / -0 | 2 |
| `src/CudaUtils.cu` | +134 / -3 | 7 |
| `src/CudaUtils.cuh` | +18 / -0 | 7 |
| `tools/memory_model.py` | +70 / -0 | 5 |

跨多个 PR 的文件需按 hunk 拆分，不能整文件搬。
