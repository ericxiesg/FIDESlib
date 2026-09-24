# 结论在你们自己的表里:CtS **失败**、StC **抛异常**——这两个是发现,不是障碍

2026-09-24,针对 `9e1ae3a`。

## 0. 先给三条能直接用的更正

**"no ModRaise test exists" —— 有。** `test/OpenFheInterfaceTests.cu:2308`,`TEST_P(OpenFHEBootstrapTest, ModRaise)`。

**"Full bootstrap A/B in the gtest framework" —— 也有。** `:3395` `OpenFHEBootstrap`,另外还有 `:3501` `OpenFHEBootstrapManualPrescale`、`:3611` `OpenFHEBootstrapLT`、`:3734` `OpenFHEBootstrapDense`。

你们 "What still needs to be done" 的 3 条里有 2 条**已经在树里了**,直接加到本次跑的测试名单即可。

**CPU@mod59/60 不要用 depth 37。** 你们自己的扫描(`82e9514`)证明了深度无关——6.0 bit 在 depth 23/30/37 完全一样。所以拿 depth 23 跑 CPU 就行,keygen 轻得多,不会再把服务器跑崩。

---

## 1. ApproxModEval "OK" **不能**为 EvalMod 洗清嫌疑

你们表里 `Expected 0.00390625` = **2⁻⁸**,也就是说 `result->GetLogPrecision()` 在那个点上报的是 **9 bit**——**CPU 参照本身就只有 9 bit 精度**。

一个参照只有 9 bit 的测试,**没有分辨 27 bit 问题的能力**。它给不出"通过"这个结论,只能给出"测不出来"。

而且它"通过"的方式也要看清楚:测得 0.0073,印出来的期望是 0.0039,**实测是期望的 1.8 倍**。它能过,是因为 `ASSERT_ERROR_OK` 断言的是 `2^(-logPrec+4)` 而印出来的是 `2^(-logPrec+1)`——**断言比它自己印的标准松 8 倍**(这个缺口我在 `report-cts-stc-review-20260923.md` §2.2 提过)。

**结论:ApproxModEval 这一项是 inconclusive,不是 exculpatory。**

---

## 2. CoeffsToSlots **两个模数都失败**,而这是这份报告里最硬的发现

| Index | Config | Max error | Expected | 超出倍数 |
|---|---|---|---|---|
| 8 | sparse (59/60) | 1.51e-12 | 5.68e-14 | **26.6×** |
| 9 | thormod (50/55) | 6.45e-10 | 2.91e-11 | **22.2×** |

你们把它记成 "pre-existing LinearTransform issue"。**"pre-existing" 不是解释,它只是在描述这件事被忽略了多久。**

而且注意这里和第 1 节的关键差别:**expected = 5.68e-14 = 2⁻⁴⁴,也就是这个测试的参照有 45 bit 分辨率。** 和 ApproxModEval 的 9 bit 不同,**CtS 测试是有能力说话的,而它说的是"错了 22–26 倍"**,约 **4.5 bit**。

4.5 bit 不等于 27 bit,我不夸大。但:

- 它在**两个模数下都存在**,比例一致(26.6× vs 22.2×),所以它是一个稳定的结构性缺陷
- CtS 在 bootstrap 里跑 **3 层**,而且 bootstrap 里还有 StC
- 这个测试是从 fresh 密文测 CtS 的;在 bootstrap 里 CtS 拿到的是 ModRaise 之后的东西,幅度大到 `K·q0/Δ` ≈ 28×32 = **896 倍**

**一个在孤立测试里就已经错 25 倍的线性变换,没有资格被跳过。**

---

## 3. StC 那个崩溃是**最强的信号**,不是待清除的障碍

```
terminate called after throwing an instance of 'lbcrypto::OpenFHEException'
  what(): poly.h:274:operator*=(): Modulus mismatch
```

`operator*=` 的 modulus mismatch 意思是:**一个明文和一个密文在不同的模数下相乘了。**

而这正是 `alignToDiagonals` 存在的唯一目的。它的注释我上次 review 时引过(`CoeffsToSlots.cu:120-126`):

> Taking the *minimum* level assumes every diagonal of a step is encoded at the same one. If they are not, dropping to the lowest leaves the higher ones truncated by `multPt` ... **The value survives, but it comes out wrong**, and wrong differently per slot because each diagonal feeds different slots.

所以:

- **抛出来的地方**,我们看到 crash
- **没抛出来的地方**,我们拿到一个**静默的错值**

这两者是同一个缺陷的两面。StC 在这组参数下**抛了**,说明 level/模数对齐的前提在这里**确实不成立**。而 bootstrap 跑完不 crash 的那些配置,走的就是"静默错值"那一侧。

**这是目前为止唯一一个直接的、非统计的证据,指向一个具体机制。** 请不要把它当成"要先修掉才能继续测"的障碍。

---

## 4. 我问了四次的那组输出,还是没有

`CheckPrecomputationShape`(`22ef07f`,`RawCiphertext.cu:1438/1449`)会打印 `[FIDESlib] bootstrap precomputation:` 开头的行,校验层数、每层对角线数、**层内 level 是否一致**、层间是否恰好降一级、层序。

按 Part23 §3.3 的结构分析,它**应当一行都不打印**。但第 3 节那个 modulus mismatch 说明这个前提在这组参数下被破坏了——**所以这几行现在极可能会打印,而它会直接指出是哪一层、差多少。**

这是零成本的(已经编进去了,启动时就跑),而且是第 3 节那个假设的直接检验。

---

## 5. 请按这个顺序跑

1. **`OpenFHEBootstrapTest.ModRaise`**(:2308,已存在)——thormod 和 sparse 两组
2. **`OpenFHEBootstrapTest.OpenFHEBootstrap`**(:3395,已存在)——完整 bootstrap 的 gtest A/B,两组。这直接回答"14→6 bit 掉在哪"
3. **启动日志里 `[FIDESlib] bootstrap precomputation:` 开头的全部行**(第 4 节)
4. **CPU@mod59/60,depth 23**(第 0 节)

第 2 和第 4 条一起,就能把"常数缺口"还是"模数依赖"定下来;第 1 和第 3 条一起,就能判断 StC 那个 modulus mismatch 是不是 27 bit 的机制。

**在这四条回来之前,不建议再扩大扫描范围。** 现在缺的不是更多数据点,是这四个具体的数。
