# 两个失败不是同一个缺陷;而且 `EvalLinearTransform` 是生产环境**不走**的路径

2026-09-24,针对 `3ad0eaf`。**在动 `LinearTransform()` kernel 之前请先看这份。**

## 0. 先认:我的二分框架写得太简单,是我引导出了这个结论

`ed031a2` §0 我写的是:

> **两个都失败** → 缺陷在共享的 `LinearTransform()` kernel / hoisted rotation 点积里,和多步无关。

**这句话缺了一个前提:它只在两个失败**量级可比**时成立。** 实际数据不是:

| 测试 | Max error | 量级 |
|---|---|---|
| `LinearTransform` /8 /9 | **0.0917** | ~2⁻³·⁴ |
| `CoeffsToSlots` /8 | 1.51e-12 | ~2⁻⁴⁰ |
| `CoeffsToSlots` /9 | 6.45e-10 | ~2⁻³⁰·⁶ |

**差了 10¹¹ 倍。**

一边是 GPU 算出一个**完全不同的向量**(你们 §2 贴的两列数根本不是同一个值的噪声版本),一边是在 45 bit 参照上**超出 22–26 倍、但仍然精确到 2⁻⁴⁰**。

**一个结构性错误和一个精度超标,不是同一个缺陷。** 我的二分把"都红了"当成"同一个原因",这个框架是我写的,结论的偏差责任在我。

## 1. LT 测试**没有**跑 ModRaise,也没有跑 EvalMod

你们 §1.1 / §1.3:

> **Bootstrap 输出(before LT)是干净的** … ModRaise + EvalMod:CPU 和 GPU 匹配(47/47 bits)

`test/OpenFheInterfaceTests.cu:2882`:

```cpp
auto raised = c1->Clone(); // FHE->EvalBootstrapSetupOnly(c1, 1, 0);
```

**`raised` 就是一个 fresh 密文的拷贝。** 真正的调用被注释掉了——正是 `e87e2f9` §0.2 说的那个未定义的 `EvalBootstrapSetupOnly`。

所以那个 47/47 是 **GPU↔OpenFHE 往返转换在 fresh 密文上的检查**,不是 bootstrap 阶段的输出。**这个测试里 ModRaise 和 EvalMod 一次都没执行。**

"ModRaise + EvalMod 是干净的,所以缺口在 LinearTransform" —— 这个推论的前半句没有数据支撑。**ModRaise 仍然是零数据。**

## 2. 更要紧:生产环境**不调用** `EvalLinearTransform`

```cpp
// Bootstrap.cu:37
bool isLT = cc.GetBootPrecomputation(slots).LT.slots == slots;
```

而 `result.LT.slots` 只在这里被赋值(`RawCiphertext.cu:1002-1005`):

```cpp
if (precom->m_paramsEnc.lvlb == 1 && precom->m_paramsDec.lvlb == 1) {
    result.LT.slots = slots;
```

**只有 levelBudget == {1,1} 时 `LT` 才会被填。** THOR 用的是 **(3,3)**,所以 `LT.slots` 保持 -1,`isLT` 恒为 false,生产走的是 `EvalCoeffsToSlots`。

而 `LinearTransform` 测试用的正是 `EvalBootstrapSetup({1,1}, {4,4}, slots)`。

**那个 0.0917 的灾难性失败,在一条 THOR 从来不走的路径上。** 它是真 bug,值得修,但**它不是那 28 bit**,而且照着它去查 `LinearTransform()` kernel,查到的可能是只在 {1,1}+dim1={4,4} 下才出现的东西。

## 3. 所以 28 bit 还是没有定位,而且空白比之前更清楚

| 阶段 | 生产是否走 | 数据 |
|---|---|---|
| ModRaise | 是 | **零**(测试注释掉,且 §1 说明 LT 测试没有代跑它) |
| EvalMod | 是 | 有,但参照只有 9 bit,**无分辨率** |
| CtS(多步) | 是 | 有:fresh 22–26×,post-bootstrap 2.6–2.9× |
| StC(多步) | 是 | **零**(CPU 侧先崩) |
| `EvalLinearTransform`(单步) | **否** | 有:0.0917,灾难性 |

**唯一有数据、有分辨率、且在生产路径上的,还是 CtS 那 22–26×。**

## 4. 建议

1. **别先去查 `LinearTransform()` kernel。** 先确认 0.0917 那个失败在 {3,3} 下是否存在——如果 `EvalCoeffsToSlots` 的误差是 2⁻⁴⁰ 级而 `EvalLinearTransform` 是 2⁻³·⁴ 级,那共享 kernel 在 {3,3} 的参数下是好的,坏的是 {1,1} 特有的东西(`LT.bStep`、`{4,4}` 的 dim1、或 `result.LT` 这条从来没人用的填充路径)。
2. **ModRaise 仍然是最大的空白,而且它在生产路径上。** `e87e2f9` §3 提的系数域判据不需要 CPU 参照也不需要 `EvalBootstrapSetupOnly`:ModRaise 在系数域无损,所以 **limb 0 必须逐 bit 不变,每个新 limb 必须 = `limb0 mod q_i`**。需要一个只读的 limb 访问器。**要我写吗?** 上一轮我问过,没收到答复。
3. **StC 也还是零。** 它的 CPU 崩溃是 OpenFHE 自己的问题,但这不妨碍**单独测 GPU StC**——不用 CPU 参照,用 CtS∘StC ≈ 恒等这个性质(两者互逆),就能在没有参照的情况下测出 GPU StC 的误差。这个测试目前不存在,但写起来不难。
