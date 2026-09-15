# 回复逐算子 trace：这份数据要重测，而且第 6 节的结论我认为反了

日期：2026-09-15。**C++ 本轮未改。**

插桩做得很好——58 步、形状、level、NoiseLevel、内存、耗时全有，这正是定位模式 B 需要的东西。
三点回应，第一点最重要。

---

## 一、这份 trace 跑在一个已知会改 level 的 bug 上，level 那一栏要重测

trace 用的是 **`2b1a029`**。而昨天那个修复是 **`acb256f`**，在它之后。

`alignToDiagonals`（我加的，为 FIXEDMANUAL 的差一 bug）在 `2b1a029` 上是**无条件执行**的，
而它就在 bootstrap 内部：

```
src/CKKS/Bootstrap.cu:112   EvalCoeffsToSlots(ctxt, slots, false);
src/CKKS/Bootstrap.cu:145   EvalCoeffsToSlots(ctxt, slots, true);
src/CKKS/Bootstrap.cu:277   EvalCoeffsToSlots(ctxt, slots, false);
src/CKKS/Bootstrap.cu:309   EvalCoeffsToSlots(ctxt, slots, true);
```

**四次，CtS 和 StC 都有。** 每一次的每个 LT step 之前，它都会把密文强制降到该 step 对角线的
level——在 FLEXIBLEAUTO 上这是没有必要的，而 bert-tiny 就是 FLEXIBLEAUTO。

所以第 3 节整节、特别是那条头号观察——

> **Level 周期**：每次 bootstrap 恢复到 L=9（不是 L=25）

——**是在 bootstrap 内部有一个会动 level 的 bug 的情况下量出来的**。
它可能就是那个 bug 的直接表现，也可能不是，但在 `acb256f` 上重测之前没法分辨。

**建议：先重测，再分析 level。** 耗时和内存那两节受影响小，结论大体可留。

## 二、第 6.3 节的噪声算术不成立，而且你们自己的实验已经证伪了它

> 每次 bootstrap 引入约 2^-25 ~ 2^-30 噪声，累积 34 次后约 34 × 2^-27 ≈ 2^-22

**bootstrap 是重置噪声的，不是往上叠的。** 它把密文重新构造出来，输出误差主要由它自己的
近似精度决定，和输入误差基本无关（只要输入还能解密）。所以不存在"34 次相加"。

而且**你们上一轮已经把这条证伪了**：`N` 从 32768 加到 65536（上一份报告的 #5），
**结果毫无变化**。如果真是累积的 bootstrap 噪声，N 翻倍（bootstrap 精度大约翻倍，按 bit 算）
一定看得出来。没有变化，就说明**瓶颈不在 bootstrap 精度**。
你们当时自己也写了这一条，这一版的结论反而退回了噪声说。

还有一层：**"解密失败"不是"精度不够"的症状。**
2^-22 的相对误差是大约 6 位有效数字——足够解码、足够分类。
OpenFHE 的 `approximation error is too high` 是在**误差估计超过明文模数**时抛的，
那通常意味着**尺度记账错了一个因子**（比如差一个 Δ），而不是丢了几位精度。

## 三、你们的发现 #5 才是线索，而且我能帮它缩一半范围

> **Bootstrap 输出 NoiseLevel=2**：不是 canonical

这条值得当主线。而且它**不是我们引入的**——查过了：

```
$ git diff --stat fa97286..HEAD -- src/CKKS/Bootstrap.cu
(空)
```

**`Bootstrap.cu` 我们一行都没动。** 所以"bootstrap 输出 NL=2"是上游行为，main 上也一样——
这和模式 B 在 main 上同样失败（上一份报告 #3）**正好吻合**。

### 一个能同时解释全部三个否定结果的机制

把它和第二节那句"尺度差一个因子"接起来：

**如果最终密文在被解密时仍处于 NoiseLevel=2（尺度 Δ²），而解密按 Δ 去除，
结果就会大 Δ 倍，误差估计随即爆掉——报出来正是 "approximation error is too high"。**

这个机制能解释你们观察到的**每一件事**：

| 观察 | 噪声累积说 | 尺度记账说 |
|---|---|---|
| main 上同样失败 | 说不通（main 也该更好些） | ✓ 上游行为，两边一样 |
| L 23→25 无改善 | 勉强 | ✓ 与 L 无关 |
| **N 32768→65536 无改善** | ✗ **被证伪** | ✓ 与 N 无关 |
| 报的是"解密失败"而非"分类不准" | ✗ 2^-22 该能解码 | ✓ 差一个 Δ 正是这个症状 |

### 怎么验（很便宜）

我上一轮加的 `GetNoiseLevel` 就是干这个的。在**解密前**读一次：

```cpp
// classifier 解密那一行之前
std::cerr << "noise level at decrypt: " << cc->GetNoiseLevel(result) << std::endl;
```

* 读出来是 **2** → 基本坐实，解决方向是解密前补一次 rescale（或让 bootstrap 返回 canonical）；
* 读出来是 **1** → 这条排除，回到第一节重测 level。

Python 侧同样一行：`engine.noise_level(ct)`。

---

## 四、下一步建议的排序（和你们的略有不同）

1. **在 `acb256f` 上重测 trace** —— level 那一栏现在不可用；
2. **解密前打一次 `GetNoiseLevel`** —— 一行代码，直接判定第三节那个机制；
3. 上面两条有结果之前，**先不要动 N、不要减 bootstrap 次数、不要降多项式次数**
   （你们的建议 1/2/5）。N 那条已经试过没用；减 bootstrap 次数在噪声说站不住之后
   反而是反向杠杆——次数少了，两次刷新之间的链更长、放大更多。
4. GELU 省 level 移植到 C++（你们的建议 3）**是对的，但它省的是 level 不是噪声**，
   和解密失败无关，可以并行做，别指望它修好模式 B。

耗时那一节的结论我完全同意，而且很有价值：**Softmax 占 34%**，
其中大头是 60 系数的 Chebyshev 倒数和重复平方——这条和 level/噪声无关，可以独立优化。
