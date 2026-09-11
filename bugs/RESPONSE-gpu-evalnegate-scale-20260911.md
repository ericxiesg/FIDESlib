# `EvalAddPt` 没坏，坏的是 `EvalNegate`

日期：2026-09-11。接 `113e854`。**C++ 未编译**（本机无 GPU），但根因是读出来的，不是猜的。

你那条 workaround 绕过去了，但**绕过去的不是 bug 本身**，而且我认为**它还没绕干净**。

## 一、根因

`EvalNegate` 走的是 `res_gpu->multScalar(-1.0)`。而 `multScalar(double)` 最终落到：

```cpp
void Ciphertext::multScalarNoPrecheck(const double c, bool rescale) {
    ...
    NoiseLevel += 1;
    NoiseFactor *= ... ScalingFactorReal ...;
    if (rescale && cc.rescaleTechnique == FIXEDAUTO)   // <-- 注意这里
        this->rescale();
}
```

而调用方是：

```cpp
void Ciphertext::multScalar(const double c, bool rescale) {   // rescale 默认 false
    ...
    multScalarNoPrecheck(c, rescale && cc.rescaleTechnique == FIXEDMANUAL);
}
```

把两段接起来，`multScalarNoPrecheck` 里的条件是 **`(rescale && FIXEDMANUAL) && FIXEDAUTO`——
恒为假**。所以在 FIXEDMANUAL 下，`multScalar` **必然把 `NoiseLevel` 抬到 2、`NoiseFactor` 乘上 Δ，
而且永远不 rescale**。

于是 **`EvalNegate(y)` 返回的密文在 Δ²**，外表看不出来。
`EvalAddPt(neg_y, pt)` 里 `pt` 是按 `noise_scale_deg=1`（Δ）编码的，
**Δ 的明文加到 Δ² 的密文上，就差一个 Δ**——你量到的 error 4.6 就是这个。

`EvalAddPt` 本身没问题，`EvalSubPt` 也没问题。**问题是「取负」在这个库里不是 scale-中性的。**

## 二、为什么你的 workaround 还没绕干净

```python
return self.cc.EvalNegate(self.cc.EvalSubPt(y, pt))
```

`EvalSubPt` 先做，两个操作数都在 Δ，这一步对；**但最后那个 `EvalNegate` 仍然把结果抬到 Δ²**。

解密是对的——`NoiseFactor` 有记账，除回去了，所以你单测能过。
但**返回的密文不是 canonical 的**，下一个把它和 Δ 明文相加的操作就会错。
`he_layernorm` 里正好有：

```python
normalised = self.add(normalised, beta[index])     # beta 是 Δ 编码的明文
out[index] = self.add(normalised, normalised)
```

`correction` 从 `subtract(ndarray, ct)` 出来（`numeric.py:269`），走完 Goldschmidt 迭代进到 `b`，
再乘 gamma、加 beta。**所以这条链上仍然有一次 Δ 对 Δ² 的加法。**

## 三、修法

新增 `Ciphertext::negate()`，逐 limb 乘 `q_i - 1`，**不动 level、不动 scale**：

```cpp
std::vector<uint64_t> minus_one(c0.getLevel() + 1);
for (size_t i = 0; i < minus_one.size(); ++i)
    minus_one[i] = cc.prime[i].p - 1;
c0.multScalar(minus_one);
c1.multScalar(minus_one);
if (c2) c2->multScalar(minus_one);
```

这不是新发明的写法：`addScalar` 处理负常数就是 `elem[i] = cc.prime[i].p - elem[i]`，
`multIntScalar` 用的就是这个 `multScalar(vector<uint64_t>&)` 重载。**-1 是整数，乘整数本来就该免费。**
`EvalNegate` / `EvalNegateInPlace` 改成调它。

（顺带：`multIntScalar` 不处理 `c2`，等惰性重线性化那批进来之后要补；我这个 `negate()` 处理了。）

## 四、这很可能就是 v6 的保真度问题

`subtract(ndarray, ct)` 在整个移植里**只有一个调用点**：`numeric.py:269`，`he_invsqrt` 里面。
也就是**只有 stage 11 和 stage 16 会走到**。而：

* stage 01–10 从不碰它——和「01–05 在 GPU 上 relRMSE 1e-8」对得上；
* stage 11–16 是**从来没跑通过**的那几级；
* 一个坏掉的 `1/sqrt` 会让 LayerNorm 输出无意义，后面全部跟着废——
  这正是 relRMSE 1.0、best-fit scale 0 的样子。

**但还是请跑 `--per-stage`。** 如果 `norm_1` 是第一个坏掉的行，那就是它；
如果更早就坏了，那就是另一回事，别被这条解释带偏。

顺带回收我昨天那句「先怀疑 `he_invsqrt` 的收敛域」：量过了，
layer 0 的方差在 [0.229, 0.605]，窗口是 [0.15, 10.0]，下界裕度 1.53 倍，**窗口本身没问题**。
详见上一份的补充。

## 五、下一跑

**要先重新编译 C++。** 编完之后 Python 那边**不用动**——你的
`EvalNegate(EvalSubPt(y, pt))` 在修好的 `EvalNegate` 下是正确且 canonical 的，
原来的 `EvalAddPt(EvalNegate(y), pt)` 也会变对，两种写法都行，不必改回去。

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
    --binary-rotations --refresh-after-dense --depth 37 --dnum 4 \
    --bootstrap-level-budget 3,3 --light-plaintext-cache 4 \
    --per-stage --device-memory
```

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CKKS/Ciphertext.{cpp,cuh}` | 新增 `Ciphertext::negate()`，scale-中性 **未编译** |
| `api/CryptoContext.cpp` | `EvalNegate` / `EvalNegateInPlace` 改用 `negate()` **未编译** |
| `../patch/` | PR2 变成 10 个文件；重建校验与前向引用校验都重跑通过 |
