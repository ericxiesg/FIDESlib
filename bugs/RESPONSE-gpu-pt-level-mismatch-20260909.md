# 回复 GPU-root-cause-pt-level-mismatch-20260909

日期：2026-09-09。本机没有 GPU，**C++ 改动一行都没编译过**。

**根因找到了，而且我认为该改的是密文那边，不是明文。** 顺带：`0 grown at runtime` 说明
`AddRotationKeys` 那个修复生效了，密钥计划这条线可以结掉。

---

## 一、这是 FIXEDMANUAL 和 OpenFHE 预计算之间的缝

诊断信息 `would read 35 limbs from pt[0], which holds 34` 把两边都说清楚了：

* **明文**：CtS 对角线来自 OpenFHE 的 `EvalBootstrapSetup`，level 由 OpenFHE 自己定；
* **密文**：`Bootstrap.cu:632` 里 ModRaise 之后是

  ```cpp
  ctxt.c0.grow(cc.L - (cc.rescaleTechnique == FLEXIBLEAUTOEXT));
  ```

  也就是说**只有 FLEXIBLEAUTOEXT 会少一层，其它技术一律长到 `cc.L`**。

我们跑的是 **FIXEDMANUAL**，所以密文到 `cc.L` = 35 limb，而 OpenFHE 给的明文是 34 limb。差一层，
正好是你看到的数字。

**这也解释了为什么 FIDESlib 自己的测试从来没撞上**：它们用默认的缩放技术，那条 `- (== FLEXIBLEAUTOEXT)`
恰好把两边对齐了。FIXEDMANUAL 是我们为了对齐 THOR 的 `he.py` 才选的，这条路上游没走过。

## 二、该改哪一边：跟 OpenFHE 一致，降密文

你给的方案 2「让 kernel 容忍明文少 limb，缺的当零」**不能用**：`out` 也是按同一个 `grid.y` 写的，
少算一层就意味着输出的最高 limb 从来没被写过，留着上一次的残留数据——不是精度损失，是错误结果。

正确做法是 OpenFHE 本来就在做的那件事：**`EvalMult(ct, pt)` 遇到 level 不等时会把密文降到明文那一层**
（`AdjustLevelsAndDepthInPlace`）。FIDESlib 的批量乘法没有这一步，它假定两边已经对齐。

所以我在 `EvalCoeffsToSlots` 里补上了这一步：**每个 LT step 开始前，把密文降到该 step 对角线的
level**（取该层所有对角线 level 的最小值）。

```cpp
const auto alignToDiagonals = [](Ciphertext& ct, const BootstrapPrecomputation::LTstep& step) {
    int ptLevel = -1;
    for (const Plaintext& pt : step.A) { ... 取最小 ... }
    if (ptLevel >= 0 && ct.getLevel() > ptLevel)
        ct.dropToLevel(ptLevel);
};
```

放在**每个 step 之前**而不是只做一次，因为 CtS/StC 每一层都有自己的对角线和自己的 level；
只对齐第一层的话，后面几层会重犯同样的错。StC 也走同一段代码，一并覆盖。

**代价是明文少的那一层 level**——本来也拿不回来，因为那一层根本没有对应的明文可乘。

## 三、我没有去改 ModRaise 的 grow

想过把 `cc.L - (rescaleTechnique == FLEXIBLEAUTOEXT)` 改成「长到明文那一层」，但没做，
理由是 ModRaise 的语义是**把模数抬到顶**，那一步的目标 level 有它自己的意义（后面的模约简依赖它），
不该被线性变换的明文反过来决定。降密文是局部的、语义明确的，也和 OpenFHE 一致。

如果实测发现降完之后 bootstrap 精度不对，那就说明这个判断错了，届时再回头改 ModRaise。
判断依据很直接：`test_stage4_bootstrap` 的精度断言。

## 三点五、加了一行日志，把最后一环钉死

我这个修复的前提是「明文那边是权威、密文该让步」。这个前提有一半是确认过的：
GPU 侧**根本不选明文的 level**——`AddBootstrapPlaintexts` 把 OpenFHE 预计算好的对角线原样搬过来，
`GetRawPlainText` 里限数就是 `GetAllElements().size()`，OpenFHE 给几个 tower 就是几个。

没确认的是 OpenFHE 那边按什么规则定这个数（`EvalCoeffsToSlotsPrecompute` 里的 `towersToDrop`）。
本机没有 OpenFHE 源码查不了，所以加了一行日志，建 context 时会打：

```
[FIDESlib] bootstrap diagonals: CtS layer 0 holds 34 limbs; a ciphertext at L=34 has 35
           (scaling technique N)
```

一行就能看出差在哪一侧、差多少。**下次跑请把这一行发我。** 如果它显示的不是差 1，
或者随 depth 变化的规律和我想的不一样，那我这个「降密文」的修法就得重新考虑。

## 四、密钥这条线可以结了

```
49 rotation keys, 16 truncated, 0 grown at runtime
```

`0 grown` 说明：

* `AddRotationKeys` 不再「有键就跳过」的修复生效了；
* `GetBootstrapKeyLevelPlan` 的逐层模型是对的（16 把截断，没有一把在运行时需要长回来）。

这两条之前都是悬着的，现在都有数了。

---

## 五、下一次运行

1. **先 `pytest`**：`test_stage4_bootstrap` 既要**通过**，也要**精度达标**——它是这次改动唯一的
   safety net。如果它过了但误差变大，请把误差数字发我，那说明降 level 的位置不对。
2. 再跑 depth=34 那组完整一层。跑通的话，这就是**第一次在 GPU 上跑完 THOR 的一层**。
3. 之后就可以开始量真正想要的东西了：per-stage fidelity（对照
   `docs/thor_port.md` 里 clear engine 的 scale 列）、单层耗时、峰值显存。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CKKS/CoeffsToSlots.cu` | 每个 LT step 前把密文降到该层对角线的 level（对齐 OpenFHE 的 `AdjustLevelsAndDepth`）**未编译** |
