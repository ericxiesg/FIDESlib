# 背景：这个仓库在做什么（给远程调试端看）

日期：2026-09-03。分支 `bootstrap-dev`，远程默认 `gitcode/bootstrap-dev`。

## 一句话

把 **FIDESlib**（CUDA CKKS，和 OpenFHE 兼容）改造成 **THOR**（CCS'25，ePrint 2024/1881，BERT-base 加密推理）
的同态引擎，替换掉 THOR 原本依赖的闭源 `desilofhe`。目标硬件：单卡 **V100 32 GB（sm_70，CUDA 12.9）**。

## 分层

```
THOR/src/thor/he.py      (1609 行，BERT 18 个 stage，只用 ~22 个引擎原语)
        |  逐 stage 机械移植，方法名照抄
python/pyfideslib/Engine  (device="cpu" | "cuda:0" 切换，numpy 进出)
        |  pybind11
python/src/bindings.cpp -> api/CryptoContext.*   (fideslib 的 OpenFHE 兼容 API)
        |  devices.empty() -> OpenFHE CPU 回退；否则走 GPU
src/CKKS/*                (FIDESlib CUDA 内核)
```

**关键设计决定**：不另造引擎抽象层。`fideslib::CryptoContext` 的每个 API 本来就有
`devices.empty() -> OpenFHE 回退`，所以它本身就是可切换引擎。CPU 路径是 GPU 路径的对拍基准
（`PYFIDESLIB_DEVICES=cpu` vs `cuda:0` 跑同一份 pytest）。

## 参数口径（当前阶段）

功能打通优先，安全性后置：`HEStd_NotSet`、N=2^16、32768 槽、scaling 50 / first 55、depth 33、dnum 3。
测试用小参数：`log_n=13, depth=12, scaling_bits=50, first_mod_bits=55, dnum=3`（见 `python/tests/conftest.py`）。
缩放技术默认 **FIXEDMANUAL**：THOR 的 he.py 显式调用 rescale / level_down，和 desilofhe 一比一对应。

**level 语义**：FIDESlib 的 level = 顶 limb 下标 = **剩余乘法层数**。OpenFHE 的 `level` 参数是**已消耗**层数。
两者在 `Engine.encrypt(x, level=depth-remaining)` 处换算，看到不一致先确认是哪一套。

## 为什么显存是主线矛盾

V100 只有 32 GB，THOR 需要 ~250 把旋转密钥。一把完整密钥（N=2^16，L+1≈30，dnum=3）120–132 MiB
→ 光旋转密钥就 ~30 GB，放不下。所以有两条省显存的主线：

1. **按 level 截断密钥**（已实现，就是当前 bug 所在）。THOR 每个 rotation 只在固定 level 用，
   密钥只需要 `id <= maxLevel` 的 Q-limb + 全部 special limb。截到 level 14 ≈ 52 MiB，省 60%。
   密钥字节：完整 `2·dnum·(L+1+K)·N·8`；截到 level m：`2·⌈(m+1)/α⌉·(m+1+K)·N·8`，α=⌈(L+1)/dnum⌉=K。
2. **LightPlaintext**（还没做，TODO T1）。权重明文按 `{level, scale, vector<int64_t> coeffs}` 存，
   用时在 GPU 上展开成 RNS 形式。没有它 stage 权重（12 层 ~110 GB）放不进去。

## 现在的进度

- step1 按 level 截断密钥、step2 THOR 原语（conjugate / multByI / multByInteger / levelReduce /
  SetRotationKeyLevels）、step3 三分量密文（lazy relinearisation）+ pybind11 + 4 个 stage 的 pytest —— 代码都在。
- CPU pytest 全过。CUDA pytest 卡在截断密钥的 GPU 路径（见 `BUGNOTE_gpu_truncated_key_grow_segfault.md`
  和 `RESPONSE-gpu-key-grow.md`）。
- 之后是 T1 LightPlaintext → T2 移植 stage_01–05 → … → T5 单层端到端。

## 需要远程做的事（一般模式）

1. `cmake -S . -B build -DFIDESLIB_INSTALL_OPENFHE=ON && cmake --build build -j && cmake --install build`
2. 跑 `examples/key-truncation` 的两个可执行文件、`pytest python/tests`（先 cpu 后 cuda:0）。
3. 编译错误优先怀疑 OpenFHE 补丁版 1.5.1.1 的 API 名字（`GetBootstrapDepth` 签名、`MultByMonomial`、
   `MultByInteger`、`Compress`、`EvalAtIndexKeyGen` + `InsertEvalAutomorphismKey`）—— 这些都是照文档写的，
   实际签名可能有出入，**直接改掉并在 report 里记一笔**即可，不用等确认。
4. 报告写进 `FIDESlib/bugs/`，一并 push。报告里请务必带上：完整的报错原文、栈回溯、
   `compute-sanitizer` 输出（如果是 GPU 崩溃），以及你实际改了哪几行。

## 口径约定

- **不要**为了让测试通过而放宽断言阈值，除非能说明数值上为什么该放宽（例如换了 scaling bits）。
- 精度断言的量级：CPU/GPU 单次运算 ~1e-6；bootstrap 在 59/60-bit scaling 下 ~1e-5。
- 崩溃类问题：先 `compute-sanitizer --tool memcheck`，再 `gdb --args`。报告里区分清楚
  **host SIGSEGV** 和 **device illegal access**（后者在 FIDESlib 里会被 `CudaCheckErrorMod` 变成
  `printf` + `exit(0)`，看起来像"静默退出"而不是崩溃）。
