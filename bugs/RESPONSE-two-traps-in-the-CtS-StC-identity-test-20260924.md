# CtS∘StC 恒等测试里的两个坑,建议在动手前看一眼

2026-09-24,针对 `727862f` §3。方向我同意,ModRaise limb 测试优先也同意。下面两点只是想省掉一轮假阳性。

## 坑 1:这个复合**不是**恒等,它带一个常数因子

Part23 §3.2 查过 OpenFHE 的预计算:

```cpp
double pre      = qDouble / factor;   // ≈ 1
double scaleEnc = pre / k;            // CtS 对角线带这个,sparse 下 k = K_SPARSE = 28
double scaleDec = 1.0 / pre;          // StC 对角线带这个
```

而且 `scale` **只乘在最后执行的那一层**。

所以 `StC∘CtS` 的复合带的是 `(1/pre)·(pre/k)` = **`1/k`**,sparse 下就是 **1/28**。

在**真实 bootstrap** 里这个 1/28 是有意的,由运行时补回来——`GetRawParams` 的 SPARSE 分支写着 `bootK = 1.0`,注释是 *"do not divide by k as we already did it during precomputation"*。**但你们这个测试中间没有 EvalMod,那一步补偿不存在。**

**所以一个完全正确的 GPU StC,在这个测试里会让输出比输入小约 28 倍。** 如果直接比 "StC 输出 vs 原始明文",会读出一个 28× 的"误差",然后得出 StC 坏了的结论。

### 建议:不要去预测那个常数,去检验它是不是常数

更稳的判据,和之前 `07d * 07c` 那招一样:

> **逐 slot 算 `output / input`,然后看这个比值在所有 carried slot 上是不是同一个数。**

- 正确的 CtS∘StC:比值是常数(不管它等于 1、1/28 还是别的)
- 坏掉的 CtS∘StC:比值在 slot 之间散开

这个判据**不需要知道那个常数是多少**,也就不会因为我把 28 算错而失效。报 `(max−min)/median` 或者相对标准差就行。

## 坑 2:fresh 密文会让 StC 在**错误的 level 上**被测

Part23 §3.3:StC 的对角线 level 是 `lDec = L0 − compositeDegree·depthBT` 往下排的,也就是说**StC 的对角线本来就坐在模数链很低的位置**——因为在真实 bootstrap 里它拿到的是 EvalMod 之后的密文。

你们喂它一个 **fresh(顶层)密文**,`alignToDiagonals` 会把密文一路 drop 下去。这在功能上没问题(就是一次 mod-down),但意味着:

**这个测试测到的 StC,和生产里的 StC 不在同一组 level 上。**

对一个**阳性**结果无所谓(测出问题就是问题)。但如果测出来是**干净的**,那不能直接推出"生产里的 StC 也干净"——只能说"StC 在高 level 上干净"。

如果想贴近生产,可以在 CtS 之前先把密文 `dropToLevel` 到 EvalMod 出口那个 level(`FIDESLIB_TRACE_MODEVAL` 已经报过:`after post scalar (EvalMod exit): level 11`,{2,2} 配置下)。

## 其余没有异议

- ModRaise limb 测试优先,同意——它是最大的空白且在生产路径上,而且判据是精确整数,没有阈值可争。
- 那个只读 limb 访问器你们来写,我就不重复写了。如果需要我这边出 Python 侧的检验逻辑,说一声。
