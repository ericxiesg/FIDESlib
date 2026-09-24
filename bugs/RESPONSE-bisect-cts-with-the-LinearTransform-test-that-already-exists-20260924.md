# 用已经存在的 `LinearTransform` 测试把 CtS 那 22–26 倍劈成两半

2026-09-24。接 `da85568`。

## 0. 一个免费的、精确的二分

两个测试都在树里,**参数只差一样东西**:

| 测试 | 行 | levelBudget | 走哪个函数 |
|---|---|---|---|
| `LinearTransform` | `:2828` | **{1,1}** | `EvalLinearTransform` |
| `CoeffsToSlots` | `:2995` | **{3,3}** | `EvalCoeffsToSlots` |

而这两个函数**最终调用的是同一个 kernel**,只是参数不同:

```cpp
// CoeffsToSlots.cu:72 —— 单步
LinearTransform(ctxt, slots, bStep, Aptr, /*stride=*/1, /*offset=*/0);

// CoeffsToSlots.cu:198-201 —— 多步
int stride = step.bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1];
int offset = step.rotOut[0];
LinearTransform(ctxt, step.slots, step.bStep, Aptr, stride, offset);
```

**单步路径永远是 `stride=1, offset=0`。多步路径传的是非平凡的 stride 和 offset。**

所以:

- **`LinearTransform` 通过 + `CoeffsToSlots` 失败** → 缺陷在**非平凡 stride/offset 的处理**,以及多步脚手架(`alignToDiagonals` 的逐层降级、层间 rescale、跨层 offset 递延)。**不在共享的 kernel 本身。**
- **两个都失败** → 缺陷在共享的 `LinearTransform()` kernel / hoisted rotation 点积里,和多步无关。

一次运行,两个已有测试,把那 22–26 倍劈成两个不相交的一半。

## 1. 为什么我认为大概率是前者,而且指向哪里

Part23 §4.1 指出 FIDESlib 相对 OpenFHE **唯一的结构性差异**就是 AFFINE_LT:OpenFHE 每层是

```
M = Σ_n diag(coeff_n) · R_{(n - offset)·sc}
```

而 FIDESlib 的 `rotIn[j] = j·sc`(**没有 `- offset`**),然后把**所有层欠下的 offset 累加起来,一次性折进最后一层的 `rotOut[0]`**。我核过这段代码,`RawCiphertext.cu:1156-1163`:

```cpp
int acc_offset = 0;
for (int i = 0; i < result.CtS.size(); i++)
    acc_offset += (result.CtS[i].slots / 2) * (...);
auto& j = result.CtS.back().rotOut[0];
j = ReduceRotation(-acc_offset, slots_red);
```

**关键:这个跨层递延只在 levelBudget > 1 时存在。** {1,1} 下 `result.CtS` 只有一层,累加退化成一项,"offset 要能穿过 M"这个要求根本不出现。

Part23 把 AFFINE_LT 排除掉的理由是:"任何旋转错位都会让输出槽位整体错排,解密是乱的,不可能还剩 10.5 bit"。**这个论证对"总体错排"成立,但 CtS 测出来的是 22–26 倍——那不是错排,是有界误差。** 所以那条排除**没有覆盖这个量级**。

我不是说 AFFINE_LT 一定错。我是说:**它是唯一已知的结构性差异,它只在多步路径里出现,而多步路径正是红的那个,而单步路径的测试就在旁边没人跑。**

## 2. 顺带:这也解释了为什么 ApproxModEval "看起来没事"

`ApproxModEval` 的参照只有 9 bit 分辨率(`Expected = 2⁻⁸`),而 `CoeffsToSlots` 的参照有 45 bit(`Expected = 2⁻⁴⁴`)。

**两个 stage 不是"一个好一个坏",是"一个测得出一个测不出"。** 在把有分辨率的那个修干净之前,没有分辨率的那个给不出信息。

## 3. 请跑(全部是已有测试,一次重建)

1. **`LinearTransform`**(`:2828`)—— thormod / sparse 两组。**这是第 0 节的二分,优先级最高。**
2. **`ModRaise`**(`:2308`)—— 第四次提;排除法里唯一完全空白的 stage。
3. **`OpenFHEBootstrap`**(`:3395`)—— 完整 bootstrap 的 gtest。
4. StC 崩溃前最后一行 stdout(`dd02758` §1,零成本)。
5. `[FIDESlib] bootstrap precomputation:` 开头的行(第六次提)。

第 1 条的结果决定接下来查 `LinearTransform.cu` 还是查 `CoeffsToSlots.cu` 的多步脚手架——这两个方向的工作量和代码位置完全不同,**不该靠猜**。
