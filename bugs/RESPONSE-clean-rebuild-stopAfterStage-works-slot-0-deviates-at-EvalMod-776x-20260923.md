# 干净重编译后 stopAfterStage 生效：slot 0 在 EvalMod 首次偏离 776x

日期：2026-09-23。针对 `e2f3506`。**`rm -rf build*` 干净重编译后重跑实验 3。**

---

## 0. 结论

| stage | slot 0 | others p50 | max | ratio | 状态 |
|---|---|---|---|---|---|
| 1 (ModRaise) | 0.00720 | 0.00545 | 0.0812 | 1.3x | ✅ 正常 |
| 2 (CoeffsToSlots) | 7.39e-05 | 0.0536 | 0.286 | 0.0x | ✅ slot 0 很小（正常） |
| **3 (EvalMod)** | **0.132** | **1.70e-04** | **1.11e-03** | **776.3x** | ❌ **slot 0 爆了** |
| 4 (SlotsToCoeffs) | 0.173 | 0.132 | 0.255 | 1.3x | ✅ 又正常 |

**slot 0 在 EvalMod 阶段首次偏离。** 之前四 stage 逐位相同是因为 `stopAfterStage` 没生效（增量构建的 ODR 违反），干净重编译后修复了。

---

## 1. 对比

### 1.1 增量构建（之前）

```
stage 1 (ModRaise     ): slot 0 0.705801   ratio 1.3x
stage 2 (CoeffsToSlots): slot 0 0.705801   ratio 1.3x    ← 和 stage 1 一样！
stage 3 (EvalMod      ): slot 0 0.705801   ratio 1.3x    ← 和 stage 1 一样！
stage 4 (SlotsToCoeffs): slot 0 0.705801   ratio 1.3x    ← 和 stage 1 一样！
```

### 1.2 干净构建（现在）

```
stage 1 (ModRaise     ): slot 0 0.00720   ratio 1.3x
stage 2 (CoeffsToSlots): slot 0 7.39e-05  ratio 0.0x
stage 3 (EvalMod      ): slot 0 0.132     ratio 776.3x   ← 偏离！
stage 4 (SlotsToCoeffs): slot 0 0.173     ratio 1.3x
```

**干净构建后四个 stage 完全不同。** 协作者的判断完全正确——增量构建导致 `stopAfterStage` 没生效，旧的三参数 `Bootstrap` 被链接了。

---

## 2. EvalMod 是什么

EvalMod 是 bootstrap 的第三步——**近似模归约**，即用 Chebyshev 级数逼近 `sin(2πKx)/(2πKx)`。它涉及：
- Chebyshev 多项式求值（多次乘法和标量乘）
- 预计算的 Chebyshev 系数
- 可能的旋转和对角线明文乘

如果 EvalMod 里的某个预计算明文在 slot 0 上的值是错的，那么**每次 bootstrap 不管输入是什么，都会在 slot 0 上出错**——这正是我们看到的签名（确定性、位置相关、数据无关）。

---

## 3. 下一步

按协作者 §3.1 的建议：**比对 CtS/StC 的对角线明文在 GPU 和 CPU 两侧的解码值，特别是 slot 0/9891/19782/20170**。

但实验 3 指向的是 **EvalMod**，不是 CtS/StC。EvalMod 里的预计算明文是 Chebyshev 系数和可能的旋转掩码——这些可能在 `ApproxModEval.cu` 里，而不是 `RawCiphertext.cu` 的 CtS/StC 对角线。

需要找到 EvalMod 里使用的所有预计算明文，在 slot 0/9891/19782/20170 上比对 GPU 和 CPU。
