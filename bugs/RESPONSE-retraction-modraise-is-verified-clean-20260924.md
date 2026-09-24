# 撤回:ModRaise limb 测试是**有效的**,我错了

2026-09-24。撤回 `6f06a7e` 和 `4825d93`。

## 0. 我错在哪

测试在每次 `store` 之前**显式做了 INTT**:

```cpp
36  GPUct1.c0.INTT(GPUcc.batch, true);   // ← 进系数域
38  GPUct1.c0.store(before);
39  GPUct1.c0.NTT(GPUcc.batch, true);    // ← 还原

45  FIDESlib::CKKS::ModRaise(GPUct1, slots, 0, true, false);

48  GPUct1.c0.INTT(GPUcc.batch, true);   // ← 进系数域
50  GPUct1.c0.store(after);
51  GPUct1.c0.NTT(GPUcc.batch, true);
```

**`before` 和 `after` 都在系数域。** 所以 `limb_i == centered(limb0) mod q_i` 正是应该成立的恒等式,PASS 是真的。

`6f06a7e` 整份报告的前提——"`store` 返回求值域"——**作为对 `store` 本身的描述是对的,但作为对这个测试的描述是错的**:调用方已经转换过了。我查了 `RNSPoly::store` 的实现,确认它不做 INTT,然后就直接下了结论,**没有回头看调用它的那几行**。

`4825d93` 那份"我把 §2(b) 自己查掉了"更糟:我很自信地关掉了自己唯一的退路,而关的方向是错的。

## 1. 为什么会漏掉

我读那段代码用的是:

```
grep -nE "ModRaise|limb|EXPECT|ASSERT|for |centered|q_i|%|EvalBootstrapSetup|store|getLevel|TEST_P"
```

**这个模式里没有 `NTT` 也没有 `INTT`。** 第 36 和 48 行因此不在我看到的输出里,而它们正是决定性的两行。

这是我这两天**第四次**同一类错误(`dropLimb` / `FIXEDMANUAL` / `ModRaise` 注释 / 这次),形状完全一样:**用一个过滤过的视图回答"这段代码做了什么",然后把看到的当成全部。** 前三次是 grep 不认注释、head 截断;这次是我自己的模式漏了关键词。

当问题是"这段代码做了什么"时,**正确做法是把整个函数读完**,不是 grep 它。这条我记下来了。

## 2. 结论:ModRaise 清白,而且是精确判据

- `after[0] == before[0]` 逐 bit 相同(系数域)
- 每个新 limb `== centered(before[0]) mod q_i`,逐系数精确相等
- sparse 和 thormod 两组都过

**没有阈值、没有参照、没有精度参数。** 这是整个调查里最干净的一次排除。ModRaise 出局。

我之前说"ModRaise 是唯一把 1 个 limb 扩成 38 个的操作,不该靠一个解释不通的 PASS 出局"——那个警惕本身没错,但那个 PASS 现在解释通了:**因为测试比我以为的做得更仔细。**

## 3. 那两个小缺口我仍然建议补(与撤回无关)

1. **只测了 `c0`。** `c1` 在 `:769/797/808` 做同样的事。判据是精确整数,复制一段就行。
2. `prescaled=true` 跳过前置块 —— 这对隔离 limb 机制是好的,只是记得覆盖的是 limb 核心而非完整语义。

## 4. 更新后的状态

| 阶段 | 状态 |
|---|---|
| ModRaise | **清白**(精确判据,本报告) |
| CtS | **红着** —— 22–26×,45 bit 参照,生产路径 |
| StC | 相对 CtS 干净,且只在 32 slot / sparse 分支(`744a883` §1–3) |
| EvalMod | 有数据,**无分辨率**(9 bit 参照) |

`744a883` 里那三条**不受这次撤回影响**:复合测试仍然分不开两个因子,CtS 那 22–26× 仍然红着,32 slot 仍然不是生产配置。**现在 ModRaise 出局了,CtS 是唯一有分辨率、在生产路径、且为红的测量。**
