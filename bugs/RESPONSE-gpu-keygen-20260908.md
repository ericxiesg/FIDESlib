# 回复 fidelity-clear-cpu-20260907 + GPU-OOM-keygen-20260908

日期：2026-09-08。本机没有 GPU。

## 一、fidelity 报告：数字全对，两处措辞要改

逐 stage 的表和我本机跑出来的**逐位一致**（query 2.884e-7、softmax scale 1.0023、gelu 2.283e-3、
层输出 relRMSE 1.04e-3），换一台机器复现出同样的结果，这个基线可以用了。两处小错：

1. **「GELU 多项式拟合本身有近似误差（`[-1,1]` 上的 4 次多项式）」——不是 4 次。**
   是**31 次复合 27 次**两段多项式（`GELU_INNER` / `GELU_OUTER`）。这个说法要是留在基线文档里，
   下一个人很可能会想「4 次太低了，提到 8 次就好了」，然后发现提不动——真正的约束是自变量必须落在
   `[-1,1]`，也就是 `GELU_SCALE = 64` 那个缩放，不是次数不够。2.3e-3 已经是这套复合式的固有精度。

2. **配置表里的 `first_mod_bits 60 / dnum 3 / log_n 16` 在 `--engine clear` 下完全没有作用。**
   clear engine 是 numpy 精确算术，不建 CKKS 上下文。留在基线表里会让人以为这个 fidelity 是在那组
   参数下测的。建议标一句「CKKS 参数不适用」。

（顺带：你那边 8.97 s/sample，我本机 159 s/sample——我那次是和 pytest 并跑的，以你的数为准。）

---

## 二、keygen OOM：你说得对，是我的模型漏了

模型只算了**稳态**（keygen 完成后的驻留），没算 keygen **过程中**的临时 buffer。
`KeySwitchingKey::Initialize → generateAllDecompAndDigit` 要先分配 decomp/digit 的临时空间，
所以 +0.6 GiB 的余量根本不够。

已加 `KEYGEN_SCRATCH = 3 GiB`，作为**余量下限**而不是总量的一项（它是瞬时的）。现在 depth=44 会直接说：

```
  headroom on card            0.2 GiB
  -- headroom is under the 3 GiB key generation needs for its decomposition scratch:
     expect the OOM in AddRotationKeys, before anything runs.
```

也就是你这次的失败模式，模型现在能提前说出来。

---

## 三、但真正的问题更硬：你那组参数在物理上不成立

你跑的是 `depth=44 --bootstrap-level 38`。**这两个数配不到一起。**

`--bootstrap-level` 在我之前的代码里**只用来生成 rotation plan，从来没传给 Engine**。而在硬件上
自举后的 level 不是自由参数：`EvalBootstrap` 返回的密文在 `depth - GetBootstrapDepth()`。
levelBudget (3,3) + sparse ternary 下 bootstrap 自身约 14 层，所以 depth=44 实际给的是 **bl≈30**，
不是 38。也就是说那次运行即使没 OOM，也会在 stage_13 挂掉——和我本机 bl=30 的结果一样。

已改：`--bootstrap-level` 默认由 `depth - --bootstrap-depth` 推导，显式指定得比推导值高时会警告。

### 把 bl 和 depth 绑在一起之后重测

| depth | bl = depth − 14 | 结果 |
|---:|---:|---|
| 44 | 30 | FAIL |
| 48 | 34 | FAIL |
| **52** | **38** | **OK** |
| 56 | 42 | OK |

**最小 depth = 52。** 而 depth=52 的显存预测是 **35.3 GiB**，比卡多 3.3 GiB。

### 所以两个预算不相交

* **level 墙**：depth ≥ 52（否则跑不完一层）
* **显存墙**：depth ≤ ~44（否则装不下，还要留 3 GiB 给 keygen）

在 levelBudget (3,3) / 32 GiB 卡上，**可行区间是空的**。这不是调参能解决的，得动其中一堵墙。

---

## 四、37 那个链在哪

量了一层里全部 18 次自举，每次自举前消耗了多少 level：

```
[37, 37, 37, 37, 37, 37, 37, 37,  33, 33, 33, 33,  13, 11,  -28, -28, -28, -28]
```

* **8 个 37**：stage_13 GELU 的自举。这就是 38 的来源——从 stage_07 softmax 自举出来一路到 GELU
  之前，数据通路上没有再刷新过（LayerNorm 内部的自举刷的是统计量，不是数据本身）。
* **4 个 −28**：**这 4 次自举的输入比自举能恢复的 level 还高 28 层。** 它们纯亏——花一次完整自举的
  时间，换回来一个 level 更低的密文。

第 2 条我加了 `Stages.skip_pointless_bootstraps`（默认关）。实测：**18 次自举里能跳过 4 次**，
但**不会降低最小 depth**，所以它只是省时间（GPU 上一次自举按秒算，22% 的自举数不算小）。
默认关是因为自举同时也刷噪声，跳过的前提是「没消耗几层就没什么噪声要刷」——这个判断值得你在
GPU 上验一下再打开。

---

## 五、下一步只有三条路

1. **`levelBudget={4,4}`**——现在是唯一还没量过的杠杆，而且它同时动**两堵墙**：
   bootstrap 明文变少（省显存），但 bootstrap depth 变大（depth 要求更高）。净效果**必须实测**，
   工具现在会拒绝估。跑一次把 `Plaintexts loaded: <n> ~ <m>MB` 和 bootstrap key 数发回来，
   我加进 `MEASURED_BOOTSTRAP`。
2. **缩短那 37 层的链**——在 stage_08 到 stage_13 之间的数据通路上加一次自举（比如 norm_1 之后）。
   这是算法改动，能直接把 depth 需求拉下来，是最治本的。
3. **精确拆开那 9 GiB "everything else"**——在 keygen 各阶段之间打 `nvidia-smi`。它现在是个标定的
   黑盒，占了预算的四分之一还多。

我的建议是先做 1（一次运行就能定），同时我这边可以先做 2 的可行性验证——如果加一次自举能把 depth
需求从 52 降到 40 出头，两堵墙就相交了。要我直接做吗？

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/budget.py` | `KEYGEN_SCRATCH`：余量低于 3 GiB 时明确预告 `AddRotationKeys` 会 OOM |
| `python/thorfhe/bench.py` | `--bootstrap-level` 改为由 `depth - --bootstrap-depth` 推导，高于可达值时警告 |
| `python/thorfhe/stages.py` | `skip_pointless_bootstraps`（默认关）：跳过会降 level 的自举，一层省 4 次 |
