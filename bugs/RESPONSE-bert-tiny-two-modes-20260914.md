# 回复：两种故障是两个主人——模式 A 是我们的回归，模式 B 不是

日期：2026-09-14。**C++ 改动未编译**（本机无 GPU）。

你把它拆成"level 下溢 crash"和"解密失败"两种模式，这个拆法是对的，而且**拆开之后归属是分开的**：
**A 是我们引入的，B 不是。**

---

## 一、模式 A 的根因不是 multPt——那个修复我们早就有了

报告的"关键发现 1"和"下一步 1"都说 bootstrap-dev 缺 `3ece51b`（multPt top-limb 修复）。
**不缺。** 查了：

```
$ git merge-base --is-ancestor 3ece51b HEAD && echo yes
yes                                   <- 3ece51b 是 bootstrap-dev HEAD 的祖先

$ grep -n "limb.at(limbsize - 1)" src/CKKS/LimbPartition.cu
695:	LimbImpl& top = limb.at(limbsize - 1);
777:	LimbImpl& top = limb.at(limbsize - 1);
783:	SWITCH(top, mult(p.limb.at(limbsize - 1)));
```

`fa97286` 那一整批（含 `1f6581d`、`3ece51b`、`7fec281`）在 2026-09-11 就 merge 进来了
（commit `9c71cdd`）。**所以"下一步 1"是个空操作，做了也不会有变化。**

## 二、真正的原因：我加的 `alignToDiagonals` 没有按 scaling technique 设限

`src/CKKS/CoeffsToSlots.cu` 里这段是我加的（修 FIXEDMANUAL 下 CtS 明文/密文差一个 limb 的 bug）：

```cpp
const auto alignToDiagonals = [](Ciphertext& ct, const LTstep& step) {
    int ptLevel = /* 这一步对角线里最低的 level */;
    if (ptLevel >= 0 && ct.getLevel() > ptLevel)
        ct.dropToLevel(ptLevel);          // 每个 LT step 之前都强制下降
};
```

注释里我自己写的动机就是 **"Under FIXEDMANUAL those disagree by a limb"**——
**但代码是无条件执行的**。而 `examples/bert-tiny/src/utils.cu:105`：

```cpp
parameters.SetScalingTechnique(lbcrypto::FLEXIBLEAUTO);
```

FLEXIBLEAUTO 自己会调 level，它的对角线本来就是配套编码的，**没有东西需要对齐**；
强行按对角线往下降，降掉的是调用方后面还要用的 level。降过头就是
`dropToLevel(-1)` → `RNSLimbs[(uint32_t)-1]` → 你看到的
`__n (which is 4294967295) >= this->size() (which is 24)`。

**这也正好解释你那张表为什么这样分布**：#2（bootstrap-dev）崩、#3（main）不崩，
而两者在 bootstrap 路径上的唯一功能差异就是这个函数。
你把变量控制得很好，结论只是归错了因。

**已改**：加了 `cc.rescaleTechnique == FIXEDMANUAL` 的前置判断，其它技术直接返回。
未编译，请重编后先跑 #2 那一格（L=23、bootstrap-dev）——**预期不再 crash**。

### 顺带，模式 A 还有第二个独立嫌疑，一条命令就能排除

我们的 level 截断密钥是**默认开**的，你的日志证明它在 bert-tiny 上生效了：

```
[FIDESlib] bootstrap key level plan: 46 of 61 keys truncated (L=25, bootstrap depth 20, ...)
```

这也是 main 没有、bootstrap-dev 有的东西，而且我们从没为 bert-tiny 的旋转模式设计过那张计划表。
上面那个修复之后如果还崩，请加：

```bash
FIDESLIB_KEY_TRUNCATION=0 ./berttiny-all
```

一次就能把它排除掉。

---

## 三、模式 B 不是我们的：你自己的 #3 已经证明了

`#3` 是 **main（`fa97286`）** 跑的，照样 0/3 解密失败。
**main 里没有我们的任何改动**，所以模式 B 与 bootstrap-dev 无关——
在我们的分支上继续找它是找不到的。

它要么是 bert-tiny 这个 example 本身的问题，要么是参数/环境。
所以报告里的"下一步 4"（问 FIDESlib 作者这个 benchmark 在哪块卡上跑通过）
是**优先级最高**的一条，其它几条都排在它后面。

## 四、三处推断我建议改掉，否则会把下一步带偏

**（1）"每次 bootstrap 引入 2^-30 噪声，累积 40 次后超出可解密范围"——bootstrap 是重置噪声的。**
它把噪声清回一个固定水平，不是往上叠。最终误差大致是
"最后一次 bootstrap 的误差 + 它之后那段电路的放大"，**不是 40 次相加**。
所以"下一步 5：减少 bootstrap 次数以减少累积噪声"很可能是错的杠杆——
减少次数反而会让两次刷新之间的链更长、放大更多。

**（2）"L 从 23 加到 25 反而更差，因为多 2 个 prime limb 让 bootstrap 初始噪声更大"——不是这个机制。**
更多 limb 不会让 bootstrap 更吵。#4 和 #3 都失败，只说明 L 不是这个问题的自变量。

**（3）"增大 N 不改善说明瓶颈不是 bootstrap 精度"——这一条我同意**，而且它比上面两条更有力：
N 从 32768 加到 65536（#5）没有变化，**基本排除了"精度不够"这一整类原因**。
这反过来支持模式 B 是逻辑/参数问题而不是噪声问题。

## 五、模式 B 我建议怎么查（在 main 上查，不在我们分支上）

和我们查 softmax 用的是同一招：**先定位到层，再定位到算子**。

现在只知道"classifier 解密失败"，中间什么样一概不知。bert-tiny 已经有
`ctxt_cpu` 和 `privateKey` 一路传进 `EvalSoftmax` / `EvalLayerNorm`（就是给调试用的），
所以可以在**每层 encoder 之后解密一次**，和明文 BERT-tiny 对一下：

* 如果第 1 层输出就已经错了 → 问题在单层内部，继续往 stage 里切；
* 如果第 1 层对、第 2 层错 → 是层间的东西（残差、bootstrap 落点、level 契约）；
* 如果两层都对、只有 classifier 错 → 就是最后那一段（`PCMM_2` + `evalTanh`），
  报告里"下一步 3"（classifier 前补一次 bootstrap）才值得试。

**在拿到这个之前，"下一步 3/5" 都是在猜。** 我们在 softmax 那个 bug 上猜错过三次方向，
每次都是逐级表把方向纠回来的。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CKKS/CoeffsToSlots.cu` | `alignToDiagonals` 限定为 FIXEDMANUAL；其它 scaling technique 不再强制降 level **未编译** |
| `../patch/` | 重新生成；重建校验与前向引用校验都通过 |
