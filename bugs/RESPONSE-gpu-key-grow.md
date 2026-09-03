# 回复：GPU level-truncated key grow segfault —— 已改，需要在真机上验证

针对 `BUGNOTE_gpu_truncated_key_grow_segfault.md`。本机没有 GPU，以下全部是静态分析 + 改写，**一行都没编译过**。

## 我核对过、确认不是原因的部分

把 `growDecompAndDigitToLevel` 和初始生成路径 `generateAllDecompAndDigit` 逐行对过，结论是
**两者产生的 limb 布局其实是一致的**，所以 bug note 里"grow 后布局和初始生成不对称"这个猜想，
按代码读下来站不住：

- key 走 `iskey=true`，此时 `generateGatherLimb` 是空操作，`bufferGATHER` 和 `bufferDECOMPandDIGIT`
  都是 `nullptr`（单 GPU 下那两处 malloc 是注释掉的），所以初始路径传给 `generate()` 的
  `buffer/offset` 实际被忽略，和 grow 传 `nullptr, 0` 等价。
- DECOMP 用 `noptr=true` + 调用方整表 memcpy，DIGIT 用 `noptr=false` 由 `generate()` 写
  `[limbs.size(), pos]` 区间 —— 两条路径一样。
- `DIGITmeta[i]` = [全部 special (id > L)] + [不属于 digit i 的 Q-limb，id 升序]，
  `DECOMPmeta[i]` = [digit i 的 Q-limb，id 升序]。两者都满足"需要的记录是前缀"，
  `lastNeededPos` 的 break 是对的。
- 内核 `hoistedRotateDotKSK_2_` 通过 `C_.pos_in_digit[i][primeid]` 索引 `DIGITlimbptr[i]`，
  通过 `pos_dec`（= partition meta 里的位置）索引 `ksk->limbptr`。grow 之后 `loadDecompDigit`
  会把 `limbptr` 整表（MAXP 项）重写，覆盖全部 id，所以那条路也是通的。
- `reloader` 捕获的是 `publicKey`（shared_ptr）**按值**，不悬空。

另外 FIDESlib 的 `CudaCheckError*` 宏在出错时是 `printf` + `exit(0)`，**不会**产生 SIGSEGV，
所以你看到的 SIGSEGV 应该是真正的 host 段错误，或者 device 非法访问把上下文打坏之后的连锁反应。

## 我改了什么

### 1. `ensureLevel` 从"操作中途"提到"操作之前"（这是我最怀疑的地方）

原来 `Ciphertext::rotate` 的顺序是：

```
in1.copy(c1); in1.modup();   // 已经往流里排了一堆 kernel，共享 aux 多项式处于 mod-up 状态
in0.copy(c0);
ksk.ensureLevel(getLevel()); // 在这里 free/malloc 密钥 limb + cudaDeviceSynchronize
in1.hoistedRotationFused(...);
```

在有 mod-up 在飞的情况下去释放/重新分配密钥显存（`GPUmalloc` 走的是全局 mempool 自由链表，
`GPUfree` 归还的块会被立刻发出去），是个很容易踩到的形态。现在 `rotate` / `conjugate` /
`rotate_hoisted`（三条分支统一在函数开头做一次预解析）都在**排任何 GPU 工作之前**先
`GetRotationKey + ensureLevel`。`relinearize` 和 `LinearTransform.cu` 本来就是这个顺序。

### 2. grow 改成"整体重建"，删掉第二套分配逻辑

`LimbPartition::growDecompAndDigitToLevel` 删除，换成
`LimbPartition::resetDecompAndDigit()`（释放 DECOMP/DIGIT limb + 把 `DECOMPlimbptr` /
`DIGITlimbptr` / `limbptr` 三张设备指针表清零）。
`KeySwitchingKey::rebuildAtLevel(m)` = `reset` + `generateDecompAndDigit(true, m)` +
`loadDecompDigit`，也就是和 `Initialize` **完全同一条路径**。贵几百微秒，但不存在第二套布局要同步。
前后各加了一次 `cudaDeviceSynchronize` + `CudaCheckErrorMod`（同步版），所以如果还有 device 错误，
会在 grow 处就报出来而不是拖到后面。

### 3. 默认不再偷偷 grow，而是抛出可诊断的异常

新增 `ContextData::allowKeyGrow`（默认 **false**）/ API `allow_key_grow` / env `FIDESLIB_KEY_GROW=1` /
`pyfideslib.Engine(..., allow_key_grow=True)`。

理由：截断密钥被用在超过声明 level 的密文上，说明调用方给 `SetRotationKeyLevels` 的
(delta -> level) 表是错的。默默重载会把一个表格错误变成每次调用一次 host→device 密钥传输
（THOR 里每层几百次），还藏在一条没人看的 warning 后面。现在默认抛：

```
KeySwitchingKey::ensureLevel: key 'xxx' (rotation index 2) was loaded truncated to level 6
but is applied to a ciphertext at level 12. Fix the level plan (SetRotationKeyLevels /
GetBootstrapKeyLevelPlan), raise ContextData::keyLevelMargin, or set FIDESLIB_KEY_GROW=1 ...
```

`examples/key-truncation` 里显式打开了 `allow_key_grow`，因为那个例子的目的就是**测量**
level plan 好不好（报告 `keys grown at runtime`）。

### 4. 新增 `KeySwitchingKey::coversLevel(level)` / `LimbPartition::decompDigitLevelCovered()`

host 侧不变式检查：某把（可能被截断的）密钥实际能服务到哪个 level。`rebuildAtLevel` 结尾断言。
调试时可以直接调它来确认"密钥到底覆盖到哪"，不用去猜。

### 5. 新的最小复现程序（不经过 Python）

`examples/key-truncation/src/key_grow_repro.cpp` → 目标 `key-grow-repro`，三个场景：

| 参数 | 场景 | 期望 |
|---|---|---|
| `inside` | 密钥声明 level 5，密文就在 level 5 用 | 精确，`grown == 0` |
| `above` | 声明 level 5，在 level 12 用，不允许 grow | 抛异常，**不许崩** |
| `grow`  | 同上但允许 grow | 重载重建，两次旋转都精确，`grown == 1` |

**注意：`inside` 这个场景之前从来没在 GPU 上跑过**（CUDA 的 stage2 在 `test_rotate` 就崩了，
`test_rotate_truncated_key_low_level` 排在它后面，根本没执行到）。所以"截断密钥在计划内能不能用"
本身还是未知数，请**先跑 `inside`**。

### 6. 测试重排

`python/tests/conftest.py` 的旋转计划改成 `{1: depth, 2: 5, -3: 3, 16: depth}`：1 和 16 是完整密钥，
2 和 -3 故意截断。`test_rotate` 只用完整密钥在顶层旋转；
`test_rotate_truncated_key_inside_plan` 在计划内用 2 和 -3；
`test_rotate_truncated_key_above_plan_raises` 断言超计划会抛（GPU-only，CPU skip）；
`test_rotate_truncated_key_grows_when_allowed` 单独建一个 `allow_key_grow=True` 的 engine 走重载路径。

这样即使 grow 路径还有问题，CUDA 的 stage2/3/4 也能跑完，不会像之前那样整条测试链断在第 5 个用例。

`test_stage4_bootstrap.py` 的 engine 也打开了 `allow_key_grow=True`：那个用例的目的是**测量**
`GetBootstrapKeyLevelPlan` 准不准（断言 `GetGrownKeyCount() == 0`），不该因为计划差一级就整个 bootstrap 报错。
如果它的 `grown` 不为 0，stderr 上会打印是哪把密钥、从哪一级到哪一级 —— 把这几行原样贴进 report，
我按那个把 `stcTop` 的公式修掉（**不要**只是加大 `key_level_margin` 蒙过去）。

## 请按这个顺序验证

```bash
cmake -S . -B build -DFIDESLIB_INSTALL_OPENFHE=ON && cmake --build build -j && cmake --install build
cmake -S examples/key-truncation -B build-kt -Dfideslib_DIR=<install>/lib/cmake/fideslib && cmake --build build-kt -j

# 1) 截断密钥在计划内到底能不能用（最重要，之前没验证过）
./build-kt/key-grow-repro inside
# 2) 超计划必须是异常而不是崩溃
./build-kt/key-grow-repro above
# 3) 重载重建路径
./build-kt/key-grow-repro grow
# 4) 三个都在 sanitizer 下再来一遍
compute-sanitizer --tool memcheck ./build-kt/key-grow-repro all
# 5) 原来的例子
./build-kt/key-truncation 13 12 25 3 3 3
# 6) pytest
export PYTHONPATH=$PWD/python
PYFIDESLIB_DEVICES=cpu    pytest python/tests -x -v
PYFIDESLIB_DEVICES=cuda:0 pytest python/tests -x -v
```

## 如果 `inside` 就崩了

那说明问题不在 grow，而在**截断加载本身**，请按这个顺序缩小：

1. `compute-sanitizer --tool memcheck` 的第一条 invalid read，记下 kernel 名和 offset。
   基本只会是 `hoistedRotateDotKSK_2_`（`rotate` 单索引和 hoisted 融合路径都用它）。
2. 该 kernel 读密钥的两处：
   - `digits[offset + dnum + i + decomp*3*dnum][decomp ? pos_dec : pos]`
     —— `decomp==false` 时是 `ksk_a->DIGITlimbptr[i][C_.pos_in_digit[i][primeid]]`，
     `decomp==true` 时是 `ksk_a->limbptr[pos_dec]`。
   - `ksk_b` 同理（`+ 2*dnum` / `+ 5*dnum`）。
   在 host 侧把 `ksk.a.GPU[0].DIGITlimb[i].size()`、`DECOMPlimb[i].size()` 和
   `cc.precom.constants[0].pos_in_digit[i][primeid]` 打出来对一遍，看是哪个索引越过了 limb 数。
   `KeySwitchingKey::coversLevel(level)` 可以直接给出"这把密钥能服务到哪一级"。
3. 如果越界发生在 `num_d`（使用的 digit 数）上：`fusedHoistRotate` 里 `num_d` 是按**密文** level
   算的（`limbsize = *level + 1`，digit i 的 start = 前 i 个 `DECOMPmeta` 大小之和）。
   截断密钥必须保证：对每个被用到的 digit i，`DIGITlimb[i]` 至少有
   `C_.pos_in_digit[i][level] + 1` 项，`DECOMPlimb[i]` 至少覆盖到 `id <= level`。
   `decompDigitLevelCovered()` 实现的就是这个判据，可以直接和实际用的 level 比。
4. 临时 workaround 依旧是 `FIDESLIB_KEY_TRUNCATION=0`（全用完整密钥）；小参数下显存够，
   不影响 T1/T2 的推进。

## 顺带

`bugs/BACKGROUND-for-remote.md` 是这个项目的整体背景（分层、参数口径、为什么显存是主线矛盾、
你在这条流水线里负责哪一段），第一次接手建议先看那个。
