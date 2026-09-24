# ModRaise limb 测试:**先别判它清白**,这个 PASS 我解释不通

2026-09-24,针对 `96b0fa4` / `f93f0ce`。**这不是说你们做错了,是说我提的那个判据被我自己写漏了一个前提。**

## 0. limb0 那一半是有效的,没问题

`after[0] == before[0]` 逐 bit 相同——这个检查正确且有意义。ModRaise 对 limb 0 做的是 `INTT` 然后 `NTT`,整数 NTT 是精确可逆的,所以往返必须逐 bit 还原。**这一半通过,是真结果。**

## 1. 但另一半:`limb_i == limb0 mod q_i` 是**系数域**的恒等式,而 `store` 给的是**求值域**

我 `e87e2f9` §3 写判据时说的是:

> ModRaise 在系数域上无损 … **每个新 limb 必须恰好等于 `limb0 mod q_i`**

**我漏了"在系数域上"这个限定对测量的含义。** 实际的 ModRaise 序列是(`Bootstrap.cu`):

```
714  ctxt.c0.INTT(...)          ← 进系数域
745  ctxt.c0.broadcastLimb0()   ← 在系数域广播 + SwitchModulus   ← 恒等式在这里成立
757  ctxt.c0.NTT(...)           ← 回求值域
```

而 `RNSPoly::store`(`RNSPoly.cpp:232`)→ `Limb::store_convert`(`Limb.cu:126`)→ `store`,**全程只是 device→host 的原始拷贝,没有任何 INTT**。所以 `store` 返回的是 **NTT(求值)域**的值。

在求值域上,这个恒等式**不该成立**:

```
after[i][j] = NTT_{q_i}(c mod q_i)[j] = Σ_k c[k]·ψ_i^{jk} mod q_i
expected    = centered(before[0][j]) mod q_i = (Σ_k c[k]·ψ_0^{jk} mod q_0) mod q_i
```

`ψ_0` 和 `ψ_i` 是**不同素数下的不同本原根**,而且外层还有一次 mod q_0 的环绕。**两者一般不相等。**

## 2. 所以这个 PASS 需要解释,不能直接当成"ModRaise 清白"

我能想到的可能:

**(a) 内层循环是空的,测试是空过的。** 这是最可能的。检查:

```cpp
for (size_t limb = 1; limb < after.size(); ++limb)            // after.size() == 1 → 整个循环不跑
    for (size_t j = 0; j < before[0].size() && j < after[limb].size(); ++j)   // 任一为 0 → 不跑
```

`RNSPoly::store` 是 `data.resize(level + 1)`。**请把测试自己打印的这两行贴回来:**

```
Before: X limbs, limb0 size=Y
After:  Z limbs
```

如果 `Z == 1`,那第二个检查一次都没执行,`newlimbs_ok` 从头到尾是 `true`——**空过**。

**(b) 我关于域的理解错了。** 有可能 `NTT()` 是幂等的 / 带 format 标志位、或者这条路径上数据本来就在系数域。如果是这样,请指出来,我收回这份报告。

## 3. 另外两个小缺口(不影响上面的主问题)

1. **只测了 `c0`,没测 `c1`。** ModRaise 对两者都做同样的事(`c1` 的 INTT/broadcast/NTT 在 `:769/797/808`)。判据是精确整数,扩到 `c1` 只要复制一段,建议加上。
2. **`prescaled=true`** 跳过了 `if (!prescaled)` 那个前置块(multScalar / rescale / dropToLevel)。**这对隔离 limb 机制是好事**,我不反对——只是要记得:这个测试覆盖的是 ModRaise 的 limb 核心,不是它的完整语义。

## 4. 正确的做法(如果 §2(a) 成立)

要在**系数域**上做这个检查,需要在 `store` 之前把多项式拉回系数域。看起来有两条路:

- 测试里在 `ModRaise` 之后显式调用一次 `ctxt.c0.INTT(cc.batch, true)` 再 `store`(注意这会改变密文状态,测试用的是拷贝所以无所谓);
- 或者在 `broadcastLimb0` 之后、`NTT` 之前插一个检查点(需要改 `ModRaise` 或加一个 debug 钩子)。

第一条更简单,而且不动生产代码。

## 5. 为什么我坚持要澄清这个

如果 ModRaise 被错误地标成清白,排除法就只剩 StC 了,接下来所有精力都会压到那一边。**而 ModRaise 恰好是那个把 1 个 limb 扩成 38 个 limb 的操作**——它是整个 bootstrap 里唯一一个"凭空造出 37 倍数据"的步骤,在一个 28 bit 缺口面前,它不该靠一个我解释不通的 PASS 出局。

---

## 附:§2(b) 我自己查掉了,不成立

我在 §2 留了一条"也许我对域的理解错了,如果是请指出,我撤回"。查完了,**这条不成立,不用你们花时间**:

- `RNSPoly::NTT`(`RNSPoly.cpp:489`)只是对每个 GPU 分区转发到 `LimbPartition::NTT`。
- `LimbPartition::NTT`(`LimbPartition.cu:483`)在 `limbsize > 0` 时直接 `ApplyNTT<algo, mode>(...)`,**没有任何 format / isNTT 标志位的短路**。

所以 `Bootstrap.cu:757` 那次 NTT 是真的执行的,`store` 拿到的确实是求值域的值。

**§1 的论证成立,§2 只剩 (a) 一种可能:内层循环没跑。** 麻烦就贴那两行(`Before: X limbs` / `After: Z limbs`),`Z == 1` 就确认了。
