# 27-bit 差距：分段 GPU-vs-CPU 的 A/B 已经写好了，就差重建

2026-09-23，针对 `cc698cf`。

---

## 0. 这份报告里最重要的一句话被埋在了倒数第二节

> The computation is deterministic (confirmed by `test_stage_stability_one_ciphertext.py` — bit-identical across runs on the same ciphertext)

**非确定性这条线就此关闭。** 不是 race，不是未初始化显存，`grow` / `generateSpecialLimbs` / `Accumulate` / `broadcastLimb0` 全部洗清。

而且是被**两条独立的路子**同时否掉的：你们的按位相等测量，和我这边的静态分析（`adaptTo` 给每个素数显式打 `U64`，guard 恒真）。两边互不依赖却指向同一结论，这个结论可以放心用。

麻烦下次把那个测试的实际输出贴一下——目前它只以一句话的形式存在，而它是接下来所有推理的地基。

---

## 1. 一个数字的澄清：6 bit 不是退步

之前记的是 10.9 bit，现在是 6.0 bit，**同一件事**：

```
σ = 0.0156
-log2(0.0156) = 6.0          ← 我写在 test 里的口径
0.0156 = 32 · 2^-11          ← 之前报告的口径
```

两者都等于 `2^-6`。`test_bootstrap_correction_factor.py` 里 `bits = -log2(rms)` 是我定的，别当成精度掉了。

**顺带一个之前被"32 ×"这个写法盖住的事实：σ 精确等于 2⁻⁶。**

累积舍入误差不会落在一个整齐的 2 的幂上。**落在 2 的幂上的是缺了一次（或多了一次）2 的幂次缩放。** 这对下面的候选排序有影响。

---

## 2. 你们的读法我要改一处，那张表比结论说的更有信息

结论写的是"larger k amplifies whatever the GPU is doing wrong"。但表里 CPU 是**平的**：

| k | CPU bits | GPU bits |
|---|---|---|
| 0 | 33.1 | 6.0 |
| 7 | 33.1 | 6.0 |
| 9 | 33.1 | 4.0 |
| 10 | 33.1 | 3.0 |
| 11 | 33.1 | 2.0 |
| 12 | 33.1 | 2.0 |
| 13 | 33.1 | CRASH |

**一个固定的相对误差被放大 2^k 之后仍然是固定的相对误差**——它会像 CPU 那样保持平坦。随放大单调劣化、最后直接崩，是**跑出可用区间**的形状：EvalMod 的近似有效区间，或者 scale / level 的裕度。

所以这张表其实给出了**两个**结论，不是一个：

1. **correction factor 不是那 27 bit 的成因**（k=0 和 k=7 都是 6.0 bit）——这条你们说对了。
2. **但 GPU 对 correction factor 的处理本身是坏的**，这是一个独立的缺陷。CPU 从 0 扫到 13 纹丝不动，说明 OpenFHE 正确地吸收了这个缩放；GPU 做不到。`Bootstrap.cu` 里 `multIntScalar(ctxt, corFactor)` 是整数标量乘、不吃 level，所以崩在 13 不是 level 耗尽，是数值出界。

第 2 条值得单独记一笔，但它不是主线。

---

## 3. 主线：27 bit 的分段 A/B 已经存在，而且已经指向我们的配置

CPU 33.1 bit、GPU 6.0 bit，**同一组参数、同一份对角线、同一个密文**。这是一个干净的 A/B，只是它现在是端到端的，需要分段。

**分段的测试早就写好了**，在 `test/OpenFheInterfaceTests.cu` 里，每一个都是拿 OpenFHE CPU 的同名函数跟 GPU 对同一个密文比、按 slot 取 max 误差：

| 测试 | 行 | 覆盖 |
|---|---|---|
| `ModRaise` | 2308 | stage 1 |
| `ApproxModEval` | 2395 | **stage 3（EvalMod）** |
| `CoeffsToSlots` | 2995 | stage 2 |
| `SlotsToCoeffs` | 3151 | stage 4 |

`22ef07f` 已经把这个 suite 指向 `tparams64_16_thor_fixmanual`（FIXEDMANUAL、depth 37、scale 50、first mod 55、sparse ternary）。**在那之前这四个测试从来没跑过我们的配置**——八组参数全是 FIXEDAUTO / FLEXIBLEAUTOEXT、depth 23。

**所以那 27 bit 掉在哪一段，重建之后一跑就知道。** 这是目前投入产出比最高的一件事，远高于继续猜。

---

## 4. 为什么我不看好你们列的三个候选

> 1. NTT/INTT correctness  2. Rescale/moddown precision  3. Key switch precision

这三个都是**通用**路径。如果它们错了，现有的 `OpenFHEInterfaceTests`（mult / rotate / rescale / keyswitch，logN 16、depth 23）应该早就红了，而它们是绿的。

也就是说：**GPU 的 NTT、key switch、rescale 在别的配置下是对的。** 那 27 bit 必然来自两边**有差异**的东西：

- **FIXEDMANUAL 专属路径**——从来没有任何测试跑过（这正是 `22ef07f` 补的洞）
- **depth 37 vs 23**
- **sparse-ternary 的 EvalMod**——`ApproxModEval.cu` 是 FIDESlib **自己实现**的，不是从 OpenFHE 搬的，degree-44 Chebyshev + `R_SPARSE = 3` 次 double-angle

第三条是我现在的首选，理由是 §1 那个 `2^-6`：double-angle 每一轮都在做 `x → 2x² - (2π)^(-2^(j-r))`，整轮下来全是 2 的幂次在动。

我读了 `applyDoubleAngleIterations`（`ApproxModEval.cu:724`）跟 OpenFHE 的 `ApplyDoubleAngleIterations` 比：

```cpp
// FIDESlib：rescale 在循环开头
for (j = 1; j <= r; j++) {
    if (FIXEDMANUAL) ctxt.rescale();
    ctxt.square(false);  ctxt.add(ctxt);  ctxt.addScalar(scalar);
}

// OpenFHE：ModReduce 在循环结尾
for (j = 1; j <= r; j++) {
    EvalSquareInPlace;  EvalAdd(ct,ct);  EvalAddInPlace(scalar);  ModReduceInPlace;
}
```

同样是 r 次 rescale，只是整体错开一位。净差别在两端：FIDESlib 的第一轮开头多 rescale 一次（吃掉 Chebyshev 出口的 degree 2），最后一轮结尾**少 rescale 一次**，所以出口是 degree 2——这正是 `72dc818` 那个 NoiseLevel=2 的来源。等价性成立的前提是入口 degree 确实是 2。`ApproxModEval.cu:53/55/110` 三个调用点的入口 degree 需要确认。

（另注：`applyDoubleAngleIterations` 的形参 `kskEval` 从头到尾没被用过。无害，但说明这个函数被改过。）

---

## 5. 请求（按重要性）

1. **重建后跑 `OpenFHEBootstrapTests`，把 `ApproxModEval` / `CoeffsToSlots` / `SlotsToCoeffs` / `ModRaise` 四个测试在 `tparams64_16_thor_fixmanual` 那一组的 `Max error` 行贴回来。** 27 bit 掉在哪一段，这四行直接给出答案。同时把原来八组的也贴上做对照——如果旧配置全绿而新配置红，那就锁定在 FIXEDMANUAL / depth 37 / sparse 这三个差异里。
2. `test_stage_stability_one_ciphertext.py` 的实际输出（§0）。
3. 启动日志里 `[FIDESlib] bootstrap precomputation:` 开头的行。按 Part23 的结构分析应当一行都没有。

暂时**不要**去动 NTT / key switch / rescale 内核——在第 1 项回来之前，那是在没有定位的情况下改通用路径，风险远大于收益。
