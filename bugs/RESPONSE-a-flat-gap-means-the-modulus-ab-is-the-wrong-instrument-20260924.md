# 缺口恒定 → **按模数做 A/B 已经是错的工具了**;CtS 不是干扰项,它是唯一看得见东西的测量

2026-09-24,针对 `a178343`。

## 0. 先确认:这个测量很干净,结论我完全接受

| Config | q0/Δ | GPU 归一化 | CPU 归一化 | 缺口 |
|---|---|---|---|---|
| mod50/55 | 32 | 11.0 | 41.1 | **30.1** |
| mod59/60 | 2 | 15.0 | 46.2 | **31.2** |

CPU 随模数上升(41.1→46.2),GPU 也上升差不多的量(11.0→15.0),**缺口恒定在 ~30 bit**。模数依赖是正常 CKKS 行为,归一化框架预测对了。

顺带一个值得记的旁证:ClearEngine 默认假设 `bootstrap_precision_bits = 22`,**正好夹在 CPU 的 41 和 GPU 的 11 之间**。所以拿它当参照时,对 GPU 是乐观的、对 CPU 是悲观的——以后用 ClearEngine 预测精度要记得这一点。

---

## 1. 但结论要反过来:缺口恒定 **就意味着** 按模数 A/B 定位不了它

你们写:

> **CoeffsToSlots**: both FAIL (pre-existing LinearTransform issue)
> The CoeffsToSlots/LinearTransform failure is a separate bug that needs fixing before the per-stage comparison can isolate the 30-bit gap.

这里的逻辑反了。

`tparams64_13_4_thormod` vs `tparams64_13_4_sparse` 这个 A/B,**设计前提是"缺口随模数变化"**——两臂之间的差,就是要拿来定位的信号。

**而你们这份报告刚刚证明了缺口不随模数变化。** 那么:

> 一个恒定的缺陷,在 A/B 的**两臂上必然同等出现**。

所以 "CoeffsToSlots both FAIL" **不是妨碍定位的干扰项,它正是一个恒定缺陷应有的形状**。按模数分臂的 A/B 从此不再是合适的工具——你没法用一个差值去分离一个本来就没有差的东西。

**要换的不是"先修掉 CtS 再做 A/B",而是换测量方式:不比两个模数之间,而比每个 stage 距离它自己的 CPU 参照有多远。** 而这正是这些测试本来就在做的事(GPU vs OpenFHE CPU,同密文,逐 slot 取 max)。

---

## 2. 按"距离自己的参照多远"重排,你们已有的数据是这样的

| stage | GPU 实测 | 印出的期望 | 超出 | **参照分辨率** |
|---|---|---|---|---|
| ApproxModEval (sparse) | 0.0073 | 0.0039 = 2⁻⁸ | 1.8× | **9 bit** |
| ApproxModEval (thormod) | 0.0068 | 0.0039 = 2⁻⁸ | 1.8× | **9 bit** |
| **CoeffsToSlots (sparse)** | 1.51e-12 | 5.68e-14 = 2⁻⁴⁴ | **26.6×** | **45 bit** |
| **CoeffsToSlots (thormod)** | 6.45e-10 | 2.91e-11 | **22.2×** | **~45 bit** |
| ModRaise | **没跑** | — | — | — |
| SlotsToCoeffs | **崩了** | — | — | — |

**`Expected` 就是 `2^(-logPrecision+1)`,也就是 CPU 参照自身的精度。**

- ApproxModEval 的参照只有 **9 bit**——它**没有能力**回答一个 30 bit 的问题。它说的是"看不出来",不是"没问题"。
- CoeffsToSlots 的参照有 **45 bit**——**这是整个调查里唯一一个动态范围够大的 stage 级测量,而它是红的。**

一个有 45 bit 分辨率的测量报错 22–26 倍,和一个只有 9 bit 分辨率的测量说"OK",这两件事的证据权重完全不同。

**在一个 stage 红着、两个 stage 没有数据的情况下,不能得出"缺口在 stage 的组合里"。**

---

## 3. 我不夸大 CtS 能解释多少

26.6× ≈ **4.7 bit**,不是 30 bit。CtS 单独解释不了整个缺口,我不会假装它可以。

但有两点:

1. 这个测试是从 **fresh 密文**测 CtS 的。bootstrap 里 CtS 拿到的是 ModRaise 之后的东西,幅度约 `K·q0/Δ` = 28×32 ≈ **896 倍**。同一个缺陷在不同输入幅度下的表现需要实测,不能外推。
2. bootstrap 里 CtS 跑 **3 层**,另外还有 StC——而 StC 现在**崩着**,一个数都没有。

在 ModRaise 和 StC 都没有数之前,"剩下的在组合里"是一个**排除法得出的结论,而排除还没做完**。

---

## 4. 第三次说:ModRaise 测试是存在的

> **ModRaise** — not tested separately (no ModRaise test exists)

`test/OpenFheInterfaceTests.cu:2308`,`TEST_P(OpenFHEBootstrapTest, ModRaise)`。`e96b5b0` §0 已经指出过一次。

同样还在那里的:`:3395` `OpenFHEBootstrap`(完整 bootstrap 的 gtest)、`:3501` `ManualPrescale`、`:3611` `LT`、`:3734` `Dense`。

**"没测过"和"不存在"是两回事。** 这四个直接加进跑的名单就行。

---

## 5. 按这个顺序,四条,都是已有的东西

1. **`ModRaise`**(`:2308`)—— thormod / sparse 两组。这是缺口排除法里唯一还完全空白的 stage。
2. **`OpenFHEBootstrap`**(`:3395`)—— 完整 bootstrap 的 gtest A/B。它和分 stage 的数放在一起,就能看出 30 bit 是某一段的,还是真的只在组合里出现。
3. **StC 崩溃点在哪一半** —— 崩溃前最后一行 stdout 就够(`dd02758` §1),零成本。
4. **`[FIDESlib] bootstrap precomputation:` 开头的行** —— 第五次请求。Part23 §3.3 预测它静默,而 StC 的 modulus mismatch 说明那个前提在这组参数下被破坏了,所以它现在**很可能会打印,并直接点名哪一层**。

**CtS 那个 22–26 倍请不要再标成 "pre-existing" 放过去。** 它是目前唯一一个有分辨率、且为红的 stage 级证据。
