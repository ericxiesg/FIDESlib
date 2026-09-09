# 状态与 TODO

日期：2026-09-04，2026-09-07 更新。分支 `bootstrap-dev`。

> **2026-09-07**：远程 agent 跑了一轮 GPU benchmark，报告在
> [GPU-benchmark-bugs-20260907.md](GPU-benchmark-bugs-20260907.md)，我的复核和修复在
> [RESPONSE-gpu-benchmark-20260907.md](RESPONSE-gpu-benchmark-20260907.md)。
> **GPU 上 stage 01 通过，之后卡在三个阻塞问题**；根因是 `MAXP=64` 常量表溢出（一个 bug 表现成三个）、
> `pcmm` 持有整个网格、掩码明文被反复编码。C++ 侧的修复**未编译**。
>
> **第二轮**（[GPU-OOM-210keys-20260907.md](GPU-OOM-210keys-20260907.md) /
> [RESPONSE-gpu-oom-20260907.md](RESPONSE-gpu-oom-20260907.md)）：`dnum=4` 证实了 MAXP 诊断，
> 但 210 把 rotation key 要 51 GiB，**截断也只降到 40 GiB**（三分之二的 key 用在满 level，无可截）。
> 解法是把旋转拆成 2 的幂：**210 把 → 15 把，51 GiB → 3.6 GiB**，代价 4.5 倍旋转次数。
> `--binary-rotations`，数值逐位不变。

下面的精度数字仍然全部来自本机 numpy 的 `ClearEngine`——精确算术加严格的 FIXEDMANUAL level/scale
契约，能证明**调度和代数正确**，不含任何 CKKS 噪声。

背景仍见 [BACKGROUND-for-remote.md](BACKGROUND-for-remote.md)。

---

## 一、单层端到端已通（T5 的主要目标）

`python -m thorfhe.bench fhe --layers 1` 跑真实 `textattack/bert-base-uncased-MRPC` 权重和真实
MRPC 样本，逐 stage 对拍明文模型：

| stage | best-fit scale | relRMSE |
|---|---:|---:|
| query / value | 2.0000 | 2.9e-7 |
| scores（stage 06） | 0.5000 | 2.9e-7 |
| softmax（stage 07） | 1.0023 | 2.3e-6 |
| attention_dense（10） | 2.0044 | 3.0e-4 |
| norm_1（11） | 1.9998 | 6.5e-4 |
| intermediate（12） | 0.0313 | 4.8e-4 |
| gelu（13） | 2.0001 | 2.3e-3 |
| output_dense（14） | 2.0000 | 2.2e-3 |
| norm_2（16） | 1.9998 | 1.1e-3 |

6 个真实 MRPC 样本（第 0 层加密、其余明文）：

| 指标 | 值 |
|---|---:|
| 层输出 relRMSE | 1.064e-3 |
| 层输出 MAE / RMSE / 最大绝对误差 | 4.50e-4 / 5.91e-4 / 5.67e-3 |
| 层输出 best-fit scale | 1.9998 |
| logits relRMSE | 6.39e-4 |
| 概率 L1（均值 / 最大） | 2.58e-4 / 1.36e-3 |
| label agreement | 100% |
| 单样本耗时（clear engine，1 层） | 159 s |

这个量级和草稿 `thor-openfhe` 的 v21 口径（RMSE ~1.3e-3、max ~9e-3）一致。

`softmax` 那个 1.0023 不是误差，是 THOR 源码末尾 `int(1/(2*D_delta))+1` 的固定轻微过归一化，
`thor-openfhe/SOFTMAX-NOTES.md` 里独立复现过同一个数。

---

## 二、这一轮定下来的四件事

### 1. THOR 的「密文恒为 2 倍」不变量

**每条密文携带的都是它所表示的值的两倍。** 权重编码器折半（`gather_upper_diagonal_batch` 里的 `/2`），
bias 在 `y + conj(y)` 加倍**之前**加进去（所以相对已经 2 倍的输入，bias 是 1 倍），LayerNorm 那个
从不抵消的加倍再把下一层放回同一基准。

之前把它误读成「QKV 输出 `x@W.T + 2b`」——那只是 1 倍输入下的表象。真正的后果是**第 0 层入口必须
喂 2 倍 embedding**。改完之后 QKV 和 attention score 直接精确到 float32。

例外只有一个：GELU。多项式拟合只在 `[-1,1]` 上有效，所以自变量必须是真实的 pre-activation/64，
不能带因子。`GeluMixin.gelu(x, carrier)` 因此在 **tanh 的自变量上除以 carrier，线性因子保持 64**，
结果仍落在 2 倍基准上。这是本移植相对 `he.py` 的一处有意偏离。

### 2. softmax 的 key 缩放是 1/64，不是 THOR 的 1/512

softmax 不是尺度不变的，`he_softmax(x) ≈ softmax(x)`，所以送进去的必须是 BERT 原样的 attention
score。实测链路给它 `4 * (q·k) * scale`（q、k 各 2 倍，stage 06 内部掩码贡献一个 1/2，stage 07 的
自举折叠再乘 2），因此 `scale = 1/64`。THOR 写 1/512，它自己的 softmax 必然在别处补了这个 8 倍。
`--per-stage` 就是复核这个常数的手段。

### 3. attention mask 是**一组**明文，每个 score 密文一个

score 以对角线形式存放：密文 `ct` 的槽 `(group, tau, block)` 携带的 key 位置是
`(ct*pack + group + tau) mod dim`。只按槽下标做的掩码只能表达「这个 query 是 padding」，而 softmax
的分母需要「这个 **key** 位置是 padding」。之前用错，分母就在全部 128 个 key 上求和——链路照样跑，
只是安静地错 33%。

`thor-openfhe/SOFTMAX-NOTES.md` 里那条「AInv 递推要求分母属于 `[epsilon,1]`，换模型必须联合标定
单点范围、每行 AExp 之和、分母上下界」正是同一件事的另一面。

### 4. 残差要显式对齐 level

跳连接停在 stage 01 的 level，分支已经花掉十几个。desilofhe 隐式对齐，FIXEDMANUAL 不会——
`LayerNormStages._residual` 现在显式 `align`。

---

## 三、Benchmark CLI

`python -m thorfhe.bench {info,fetch,reference,fhe}`。目标机有 proxy，所以：

* `--proxy`（或 `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY`）、`--hf-endpoint`（镜像站）、`--token`
* `--cache-dir`（或 `THORFHE_CACHE`，默认 `~/.cache/thorfhe`）、`--offline`
* 先 `fetch` 暖缓存，之后测量运行不碰网络

目标机只有 numpy + requests，所以下载器、checkpoint 读取（safetensors + torch zip + **pre-1.6 legacy
pickle**，受限 Unpickler，白名单外的 global 一律拒绝）、WordPiece 分词器、numpy BERT 参考实现全部
自己写，不引 torch/transformers。

明文路径已验证：64 条 MRPC validation 上 **84.4% / F1 88.6%**，与该 checkpoint 公开数字一致。

指标分三类，不要混：**accuracy**（对数据集标签）、**fidelity**（MAE/RMSE/最大绝对误差/relRMSE，
对明文模型）、**probability**（softmax L1、label agreement、翻转样本的最小 margin）。
一次运行可以 100% 忠实而只有 70% 正确，也可以完全不忠实却碰巧正确。

`--per-stage` 逐级解密并报告 best-fit scale + 残差，是定位发散点的工具——上面四个发现全是它找出来的。

---

## 四、TODO

### 立刻能做（不需要 GPU）
- [ ] **12 层全加密**。目前只跑过 1 层，`--layers 12` 没试过。level 预算是主要风险：stage 12–14 要
      17 个 level（GELU 占 13），真实自举之后够不够没验证过。
- [ ] **整个 validation set 的 accuracy**。现在只有 1 个样本的 fidelity 和 64 个样本的明文基线。
      clear engine 上单层约 131 s/样本，12 层 × 408 样本不现实——需要先定一个抽样口径。
- [ ] **stage 17/18 接进 `bench`**。pooler + classifier 已经实现并单测过，但 `run_encrypted` 目前
      在解密后走明文头。
- [ ] `--calibrate` 是否真的生效没确认过：两次运行数字逐位相同，但打印出来的窗口确实不同
      （`[-10.7, 10.1]` vs THOR 的 `[-27.2, 21.7]`）。THOR 自己的表现在已经够用（softmax 2.3e-6），
      所以优先级低，但这个不一致本身要查清楚。
- [ ] `encode weights` 每层约 64 s（6144 个 slot vector ×2），12 层就是 13 分钟。需要缓存到磁盘。
- [ ] `plan_rotation_keys(scope="layer")` 一次约 97 s，也该缓存。

### 需要 GPU（2026-09-07 起有了实测）
- [x] `--engine fideslib` 跑起来了，`Engine` 原语齐全（远程补了 `add`/`subtract` 的
      `np.ndarray` 分支）。stage 01 通过。
- [ ] **拉上 2026-09-07 的修复重跑**，尤其把 `MAXP` 检查抛出的 `q / p / total` 三个数字发回来，
      好定出准确的 depth 上限表。先试 `dnum=4/5`——不改 C++ 可能就解掉 Bug 1 和一半的 Bug 3。
- [ ] **Bug 2：bootstrap 运行时崩溃**，仍未定位。报告里写的 stage_02 应为 stage_03
      （`make_rotated_copies` 只有 rotate，没有 multiply）。
- [ ] **用 `--binary-rotations` 重跑**：15 把 key（3.6 GiB）+ bootstrap 22.3 GiB ≈ 26 GiB，
      在 32 GiB 卡上装得下。这是目前唯一能让整层的密钥放进去的办法。
- [ ] 中间档：给最常用的少数索引留专用 key、其余拆开，用 4.5 倍旋转换回一部分。
      接口已留在 `Stages.rotation_steps`。
- [ ] `levelBudget={4,4}`（原 T7）现在有了具体动机：bootstrap 的 10.4 GiB **明文**是第二大占用，
      减少线性变换的 giant-step 数能把明文和 key 一起压下来。
- [x] 真实自举之后要剩多少 level。**最小 bootstrap_level = 38**（2026-09-08 实测，与 depth 无关：
      depth=50/60/90 在 bl≥38 都通过，bl=36 都失败）。瓶颈是 softmax 自举到 GELU 之间那 38 个
      level。之前记的「30 不够」用的判据不可靠，已更正，见 `RESPONSE-gpu-oom-20260908.md`。
- [x] **GPU 上 stage 01–05 已验证**（2026-09-08，`report/fidelity-gpu-fideslib-20260908.md`）：
      query relRMSE 1.03e-8、scale 1.0000，64 次旋转由 15 把二进制密钥完成。引擎原语、light
      plaintext、binary rotations、pcmm 在硬件上都是对的；未验证的只剩依赖 bootstrap 的部分。
- [x] **runtime grow OOM 的根因**：`GPUmalloc` 是按精确字节大小分类的 slab 池，默认 slab **1 GiB**，
      而且 slab **从不还给 driver**。所以「8.4 GiB 空闲」大多躺在别的 size class 的空闲表里。
      已加自适应减半重试（**未编译**），见 `RESPONSE-gpu-runtime-grow-20260908.md`。
- [x] **`AddRotationKey` 用 `std::map::emplace`，索引重复时静默丢弃后来的键**（2026-09-09）：
      `SetRotationKeyLevels` 和自举预计算会请求同一个索引（binary rotations 下两边都是 2 的幂，
      必然重叠），先到的截断键留下，自举要的完整键被丢掉 → `ensureLevel` 抛错。已改成保留覆盖
      更高 level 的那把。
- [x] **illegal memory access 已定位到 `LimbPartition::LTdotProductPtBatch`**（远程用
      `CUDA_LAUNCH_BLOCKING=1` 钉住，2026-09-09）：kernel 用 `out[0]` 的 level 定 `grid.y`，
      却索引 out/in/pt **三个**数组的 limb，其中 `pt`（自举预计算明文）按自己的 level 生成，
      最可能对不上。已加逐数组边界检查，抛出会指名是哪个数组、差多少。
- [x] **`AddRotationKeys` 有键就整个跳过**：所以上一轮在 `AddRotationKey` 里做的「保留覆盖更高
      level 的键」根本没被调用到。已改成「已有的键覆盖不够就重新生成」。顺带：`GetRotationKey`
      不像 `HasRotationKey`/`AddRotationKey` 那样归一化负索引，已统一。
- [ ] **让空 slab 回到 driver**：真正的解法，需要记录 slab 基址 + 全空检测。没 GPU 验不了。
- [x] **两堵墙已经相交**（2026-09-08）：在 stage 10 之后插一次自举
      （`--refresh-after-dense`，`LayerNormStages.refresh`），把 37 层的链切成 19+18，
      **最小 depth 52 → 34**，显存 35.3 → **27.4 GiB（余量 4.6 GiB）**。
      代价是 18 次自举里多 4 次；数值上是恒等变换，实测 logits 逐位不变。
- [ ] ~~**level 预算和显存预算目前不相交**~~：bl≥38 意味着 depth≈50，而 depth=50 预测要 34.4 GiB，
      比 32 GiB 卡多 2.4 GiB。(4,4) 实测之后：bootstrap_depth 从 14 涨到 18，level 墙推到
      depth≥56，但显存降到 23.8 GiB——**depth=51/(4,4) 只差 0.8 GiB**，是目前最接近的一组。
- [ ] 旋转密钥预算。新 stage 的索引和 level 已量过：stage 12/14 各 12 个（`±1..±5`、`±8`、2048），
      stage 17+18 共 28 个（加 `±16..±1024`、4096、8192、16384），和 01–05 的集合大部分重叠。
- [ ] `block_diag_2` 掩码族是**模 8 窗口 6**，FF 和 pooler 共用。GPU 侧若按单一 `n_slot` 公式生成
      掩码，一半会静默算错。

### 更远
- [ ] T6 显存：`SetRotationKeyLevels` 用上那张 (delta, level) 表；种子压缩密钥 `a`。
- [ ] T7 算法层：OverModRaise（ePrint 2025/1298）省 2–3 level；levelBudget {4,4} 减密钥数。
- [ ] T8 128-bit 安全参数。`thor-openfhe/PARAMS.md` 和它的 48-bit 两轮安全档可以直接参考。

---

## 五、已删除的 note

以下都已修复并验证，文件已删，改动本身在 git 历史里：

* GPU 截断密钥 grow 段错误（`KeySwitchingKey::cc` 是悬垂引用）——远程 compute-sanitizer 三个场景全过
* `LightPlaintextImpl::Load` 留下 `uid = 0`，磁盘读回的权重全部别名到缓存第一项
* `~CryptoContextImpl` 抹掉 OpenFHE 全局静态密钥表
* `SetDevices(devices)` 左值 / `std::move`
* step3 的 5 个构建与接口 bug
* T1 light plaintext（NTT 输入次序不需要 bit reverse）、T2 stages 01–05、T3 he.py stage_06
  accumulator bug、FIXEDAUTO 实验——结论都已写进 `docs/thor_port.md` 和 `docs/light_plaintext.md`
