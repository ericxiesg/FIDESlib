# 单步 LT 为什么是灾难性的:AFFINE_LT 的补偿 offset 在那条路径上**根本没被算过**

2026-09-24。读完了 `LinearTransform.cu:63-230` 和 `RawCiphertext.cu:1002-1032` 的完整函数(不是 grep——见 `2ff23dc`)。

## 0. 结论

`AFFINE_LT` 把每层欠下的旋转 offset 递延到最后一层的输出侧。**多步路径算了这个补偿,单步路径没算。**

这解释了 `3ad0eaf` 里那个 0.0917——为什么单步 LT 不是"精度差一点",而是**算出一个完全不同的向量**。

## 1. 证据

`#define AFFINE_LT true`(`BootstrapPrecomputation.cuh:8`),所以两条路径都在 AFFINE_LT 模式下。

**多步路径**(`RawCiphertext.cu:1156-1163`)——把所有层的 offset 累加,折进最后一层的 `rotOut[0]`:

```cpp
int acc_offset = 0;
for (int i = 0; i < result.CtS.size(); i++)
    acc_offset += (result.CtS[i].slots / 2) * (result.CtS[i].bStep > 1 ? result.CtS[i].rotIn[1] : ...);
indexes.emplace_back(ReduceRotation(-acc_offset, slots_red));
auto& j = result.CtS.back().rotOut[0];
j = ReduceRotation(-acc_offset, slots_red);     // ← 这个值后来成为 LinearTransform 的 offset 参数
```

**单步路径**(`RawCiphertext.cu:1002-1017`)——整个分支里**没有 `acc_offset`、没有 `rotOut`、没有任何 offset**:

```cpp
if (precom->m_paramsEnc.lvlb == 1 && precom->m_paramsDec.lvlb == 1) {
    result.LT.slots = slots;
    result.LT.bStep = ...;
    for (int i = 1; i < result.LT.bStep; ++i)
        indexes.push_back(ReduceRotation(i, slots_red));
#if AFFINE_LT
    indexes.push_back(ReduceRotation(result.LT.bStep, slots_red));   // 只有巨步那一个索引
#else
    ...
#endif
}                                                                     // ← 结束,没有 offset
```

而调用点(`CoeffsToSlots.cu:72`)**直接传字面量 0**:

```cpp
LinearTransform(ctxt, slots, bStep, Aptr, /*stride=*/1, /*offset=*/0);
```

在 `LinearTransform` 里(`:183`),`offset == 0` 走的是 `else`——**一次旋转都不做**:

```cpp
} else if (offset != 0) {
    ...
    results[j]->rotate(offset);      // 多步路径走这里
} else {
    if (results[j]->c1.isModUp())
        results[j]->modDown(false);  // 单步路径走这里:只 moddown,不旋转
}
```

## 2. 为什么这和 Part23 §4.1 对得上

Part23 指出 OpenFHE 每层是

```
M = Σ_n diag(coeff_n) · R_{(n − offset)·sc},    offset = 2^layersCollapse − 1
```

而 FIDESlib 的 `rotIn[j] = j·sc`(**没有 `− offset`**),差的那一份必须在输出侧补回来。多步路径补了;**单步路径既没在输入侧减、也没在输出侧补**,所以整个变换被旋转了 `offset` 个槽位。

一个整体旋转的输出正是 `3ad0eaf` §2 贴出来的形状——**不是同一个值的噪声版本,是一组不同的数**。

## 3. 这**不是**那 28 bit,但值得修

- `EvalLinearTransform` 只在 levelBudget == {1,1} 时被调用(`Bootstrap.cu:37` 的 `isLT`,而 `result.LT` 只在 lvlb==1 时填充)。**THOR 用 (3,3),不走这条路。**
- 所以它解释的是 `3ad0eaf` 那个 0.0917,不是生产里的 28 bit。**这也再次说明 `118e44c` §0 的判断是对的:两个失败是两个不同的缺陷。**

**修法**:在 LT 分支里按多步路径同样的方式算一个 `acc_offset`,存进 `result.LT`(需要给它加一个字段),然后 `EvalLinearTransform` 把它传给 `LinearTransform` 而不是 0。

具体数值我不写死——多步的公式是 `(slots/2) * rotIn[1]`,单步的 stride 是 1,但 `LT.slots` 和 `CtS[i].slots` 的语义不一定一样(一个是总槽数,一个是那层的对角线数)。**这个值应该推出来,不该照抄。** 我在没有编译器的情况下给一个具体常数,风险大于收益。

## 4. 我读代码时顺带注意到的两个 release 失效断言

`LinearTransform.cu`:

```cpp
:65   assert(pts.size() >= rowSize);
:76   assert(pts[0]->c0.getLevel() == ctxt.getLevel());
```

第二个正是 `alignToDiagonals` 的契约。目前 `CheckPrecomputationShape` 报告 shape 干净、09-15 的诊断也从未触发,所以**这两个现在应该都成立**——但它们在 release 下不存在,而违反第二个的后果是 `dotProductPt` 静默截断(`CoeffsToSlots.cu:120` 描述的那条路)。和 `dd02758` §2 那个静默 clamp 一样,建议改成 throw。

## 5. 对 28 bit 没有推进

老实说:这份报告把 `3ad0eaf` 那个失败解释掉了,但它在一条生产不走的路径上。**CtS 那 22–26× 仍然是唯一有分辨率、在生产路径、且为红的测量**,我读完 `LinearTransform` 没有找到能解释它的东西。
