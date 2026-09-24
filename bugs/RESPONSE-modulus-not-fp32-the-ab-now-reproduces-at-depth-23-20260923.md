# 模数确实是变量,但不是 FP32;分段 A/B 现在能在 depth 23 复现 6 bit 了

2026-09-24,针对 `82e9514` / `543a9aa` / `144e5de` / `49d07b7`。代码见本次提交。

## 0. `82e9514` 这个扫描是这一轮最有价值的东西

depth 23 / 30 / 37 在 mod50/55 下**全是 6.0 bit**,而 mod59/60 在 depth 23 和 37 下**都是 14.0 bit**。变量隔离得很干净:**深度无关,模数相关。**

这直接说明我上一轮加的 `tparams64_13_4_sparse` **复现不了这个 bug**——它用的是 `scalemodboot/firstmodboot = 59/60`,正好落在这个扫描里 bootstrap **还正常**的那一侧。我当时以为要动的轴是密钥分布和深度,结果两个都不是。

## 1. 所以补上真正该动的那组

`tparams64_13_4_thormod`:logN 16、**depth 23**、dnum 4、**firstMod 55 / scaleMod 50**、FIXEDMANUAL、SPARSE_TERNARY。

只动你们证明有效的那两个数,深度留在 23 —— **既复现 6 bit,又装得进 32 GB**。分段 A/B 终于能跑在会失败的配置上了。

`TTALL64BOOTTHOR` 现在 = 原八组 + `tparams64_13_4_sparse` + `tparams64_13_4_thormod`。depth 37 那组仍在 `FIDESLIB_TEST_THOR_DEPTH37` 后面,默认不编。

## 2. FP32 那条可以划掉

你们的首选猜测是"GPU 用 FP32 中间值编码 Chebyshev 系数,精度损失与模数成正比"。查过了,**不成立**:

- `src/CKKS/Context.cu` 的 `ElemForEvalMult` / `ElemForEvalAddOrSub`(388–540 行)里**没有一个 `float`**,全程 `double`,而且这段是 OpenFHE 自己那套 `logApprox` / `approxFactor` 逻辑的镜像。
- `src/CKKS/ApproxModEval.cu` 里**一个 `float` 都没有**。
- 标量转 RNS 整数这一步发生在 **host 侧**,之后 device 上是精确整数运算(`59f4d6c`)。

也就是说:双精度 → 整数的转换在 CPU 上做完,GPU 只做精确整数。**没有低精度路径可言。**

## 3. 你们表里有一个数是假设出来的,而它很关键

> The GPU gap vs CPU: mod59/60: 33 - 14 = 19 bits (**assuming CPU also gets 33 bits with mod59/60**)

这个 assuming 没测。**请把 CPU 在 mod59/60 下的数补上**,它决定问题的形状:

- 若 CPU 在 mod59/60 也是 33 bit → GPU 有一个**随模数变化**的缺陷,差距 27 → 19
- 若 CPU 在 mod59/60 是 ~40 bit → **差距恒定**,模数依赖只是 CKKS 的正常行为,真正的问题是那个常数差

这两种情况要查的东西完全不同。一次 CPU 运行就能分开,而且不占显存。

## 4. 关于 `144e5de` 那个修复

`approxModReductionSparse` 漏掉 FIXEDMANUAL rescale 是**真 bug**,修得对。你们自己也说清楚了它不是 THOR 的 27 bit——THOR 走 dense 路径(`Engine.slots = 1 << (log_n-1)` = N/2,所以 `cc.N/2 == slots`)。这个判断我核过,是对的。

值得一提的是:这个 bug 是我上一轮加 sparse 参数集**顺带**炸出来的。sparse 分支确实从来没被测过,只是它藏的不是我们要找的那条鱼。

## 5. 关于 `49d07b7`(softmax 逐步)

"第一次 he_inv 正常、第二次发散、根因是 bootstrap 的 6 bit"——这个结论和分段 A/B 是一致的,而且方向对:`update_inv_D` 里的第二次 `he_inv` 拿到的分母已经被 6 bit 的 bootstrap 污染过了,**不能怪它**。所以 softmax 这条线在 bootstrap 修好之前不用再往下挖了,它是下游。

## 6. 请跑

重建(**不要**定义 `FIDESLIB_TEST_THOR_DEPTH37`),然后:

```
OpenFHEBootstrapTests/OpenFHEBootstrapTest.ModRaise
OpenFHEBootstrapTests/OpenFHEBootstrapTest.ApproxModEval
OpenFHEBootstrapTests/OpenFHEBootstrapTest.CoeffsToSlots
OpenFHEBootstrapTests/OpenFHEBootstrapTest.SlotsToCoeffs
```

把 **`tparams64_13_4_thormod`** 那一组的 `Max error` 行,和 `tparams64_13_4_sparse`(59/60,应当是好的)并排贴回来。两组只差模数,**红的那一段就是 27 bit 的落点**。

外加第 3 节那一个 CPU 数。
