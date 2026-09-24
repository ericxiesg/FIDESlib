# 我有两条是错的:ModRaise 测试是注释掉的,StC 崩溃在 CPU 侧

2026-09-24,针对 `5900660`。

## 0. 先认两个错

### 0.1 ModRaise 测试:我说了三次"它存在",这是错的

`e96b5b0` §0、`da85568` §4、`ed031a2` §3 三次说 ModRaise 测试在 `:2308`。**源码里那行确实在,但整块被 `/* */` 包着**——`:2306` 开,`:2394` 闭。它不在 `--gtest_list_tests` 里,跑不了。

我的证据来自 `grep -n "TEST_P(OpenFHEBootstrapTest,"`。**grep 不认识 C 的块注释。** 我拿一个 grep 命中当成了"这是活代码",而且据此让你们跑了三次。

这和之前 `dropLimb`、`FIXEDMANUAL` 两次是同一类错误:**把文本匹配当成语义证据。** 抱歉,浪费了你们的时间。

### 0.2 但**先别直接取消注释**——它链接不过

`:2361` 用的是:

```cpp
auto raised = FHE->EvalBootstrapSetupOnly(c1, 1, 0);
```

Part23 §2 查过:`EvalBootstrapSetupOnly` / `EvalBootstrapNoStC` / `EvalBootstrapDensePartial` 这三个只在**未启用**的那份补丁(`openfhe-1.5.1.patch`)里有**声明、没有定义**。实际用的是 `fideslib-ref-1.5.1.1.patch`。

旁证:`CoeffsToSlots` 测试里那行是 `auto raised = c1->Clone(); // FHE->EvalBootstrapSetupOnly(c1, 1, 0);`——**同一个调用,被注释掉换成了 Clone。**

所以这个测试大概率就是因为这个才被整块注释掉的。取消注释会直接链接失败。**这也是为什么它一直没人跑,而不是被遗忘了。**

### 0.3 StC 崩溃:我的解读也是错的

`e96b5b0` §3 我把那个 modulus mismatch 读成"GPU 侧 level/模数对齐前提被破坏的直接证据",还说它是"目前唯一一个直接的、非统计的、指向具体机制的证据"。

**`5900660` §2 证明它在 CPU 侧**:`Run SlotsToCoeffs` 打印了,`FHE->EvalSlotsToCoeffs()` 抛异常,`Run SlotsToCoeffs GPU` 从未打印。GPU StC 根本没执行过。

这条线整个作废。`dd02758` §1 那个"看最后一行 stdout"的定位方法是对的,而它给出的答案否定了我的假设——这正是它存在的意义。

## 1. `CheckPrecomputationShape` 静默 = shape 正确,这条干净地关掉了

你们 §4 报告它一行都没打印,并且正确地推断了含义:每层对角线同 level、层间恰好降一级、第一层不高于密文 level。

**这和 Part23 §3.3 的结构分析完全吻合**(同一层 63 条对角线共用一个 `paramsVector[s]`,层间逐层 erase 一个 Q)。**分析和实测两条独立的路径给出同一个结论,这个假设族可以彻底关掉了。**

这也是 `22ef07f` 那个 validator 的用处——它的价值不在于报警,在于**它保持沉默时,沉默是有意义的**。

## 2. 你们 §3 那个结论,我要请你们再看一眼

> **CtS 不是 GPU 特有问题**——CPU 和 GPU 在同一个输入上精度相同(42=42, 33=33)。

"CPU CtS bits 42 / GPU CtS bits 42" 这两个数,如果是分别对两个解密结果取 `GetLogPrecision`,那它们是**两个自估**,不是一个比较。两边各自"看起来"有 42 bit 精度,和"两边算的是同一个值"是两回事。

真正的比较量是你们同一行里的 **Max error 1.31e-12 对 Expected 4.55e-13 = 2.9×**。这是 GPU 相对 CPU 的实际差异,它不为零。

而且有一个数需要解释:**同一个 CtS,独立测试里是 22–26×,这里是 2.6–2.9×。** 两者差了近十倍。你们指出了输入不同(fresh vs post-bootstrap),这个观察是对的——**但它是一个待解释的现象,不是一个结论。** 同一段代码在两种输入下偏离参照的程度差十倍,本身就是线索。

我不坚持 CtS 是 28 bit 的来源(4.7 bit 也好 1.5 bit 也好,都不是 28)。我只是说这两个数不该被"CtS 没问题"一句带过。

## 3. ModRaise 怎么测:不需要 `EvalBootstrapSetupOnly`

ModRaise 的正确性有一个**不需要任何 CPU 参照**的判据,因为它在系数域上是**无损**的:

1. `INTT` → 系数域
2. `grow(cc.L)` → 扩出 37 个 limb
3. `broadcastLimb0()` → 把 limb 0 复制到每一个新 limb,逐个 `SwitchModulus`
4. `NTT` → 回到求值域

所以:**ModRaise 之后,limb 0 必须与之前逐 bit 相同;而每一个新 limb 必须恰好等于 `limb0 mod q_i`。**

这是一个精确的整数判据,没有精度、没有参照、没有阈值。它直接检验 `grow` / `broadcastLimb0` / `SwitchModulus` 这三步——也就是排除法里唯一完全空白的那一段。

**但目前做不了:** 绑定里只有 `GetLimbTableSizes`(只给表的长度),没有任何暴露 limb **内容**的接口。要做这个检查需要加一个只读的 limb 访问器。

我没有直接写它,因为这台机器没有编译器,而一个我无法编译、无法运行的新 C++ 接口交给你们,风险大于收益——上面 §0 刚发生过两次我没核实就断言的事。**如果你们认为值得,我可以写;或者你们直接加一个 `GetLimbValues(ct, index)` 更快。** 请说一声。

## 4. 现在的状态

| 阶段 | 有数据吗 | 说明 |
|---|---|---|
| ModRaise | **没有** | 测试被注释,且依赖未定义的 `EvalBootstrapSetupOnly` |
| CoeffsToSlots | 有,但矛盾 | fresh 输入 22–26×,post-bootstrap 输入 2.6–2.9× |
| ApproxModEval | 有,但无分辨率 | 参照自身只有 9 bit |
| SlotsToCoeffs | **没有** | CPU 侧先崩了,GPU 从未执行 |
| 完整 bootstrap | 有 | 恒定 28 bit 缺口(32 slots),与 Python 的 ~30 bit 一致 |

**四个 stage 里两个完全没有数据。** 在这种情况下,"缺口是整条管线累积的"仍然是一个**排除法结论,而排除没有做完**——这一点和 `da85568` §3 说的一样,只是现在更清楚了:空白的是 ModRaise 和 StC,而这两个恰好是 EvalMod 两侧的邻居。
