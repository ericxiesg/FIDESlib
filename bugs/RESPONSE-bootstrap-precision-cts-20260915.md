# 判据结论成立。再给一条独立证据、排掉一个方向、以及一个我们自己的嫌疑

日期：2026-09-15。针对 `34460bf`。

---

## 1. 结论成立

`q0/Delta` 翻了 4 倍（32 → 128），绝对误差从 2.27e-2 到 2.22e-2 —— 没动。
如果误差来自 Chebyshev/double-angle，它相对 `q0` 是固定的，换算到消息域就该跟着
`q0/Delta` 涨到 ~0.09。**没涨，所以不在模约简那一侧。** 判据干净，结论采纳。

---

## 2. 还有一条独立证据，指向同一个结论

你第一次的三个值本身就说了同一件事，只是当时没往这边看：

```
值      =  2.0        -3.0        0.5
误差    = +4.67e-3    +1.11e-3   -1.70e-2
相对误差=  0.23%       0.037%     3.4%
```

**误差不是值的函数** —— 最小的那个值（0.5）误差最大，是 −3.0 的 **15 倍**；
而且符号也不一致。既不随 `|v|` 涨，也不随 `|v|` 跌成比例。

这一条很有分量，因为**模约简那一侧的误差必然是 `m/q0` 的函数**（正弦在
`m/q0` 处的近似误差），也就是必然随值变化。它不随值变化 ⟹ 不在那一侧。
这和第 1 节是两条独立的证据，结论一致。

反过来，**它随槽位变化**——三个相邻槽差 15 倍。那正是线性变换（CtS/StC 是 DFT
那一类）会有的形状。

---

## 3. 你列的下一步 #3（correctionFactor）可以排掉，用你自己的数据

> "correctionFactor = 0 时 `correction = correctionFactor - deg` 是负数，
> `corFactor = 1 << llround(correction)` 会产生奇怪值"

`corFactor` 是最后乘上去的一个 **2 的幂**（`Bootstrap.cu:318` 的 `multIntScalar`）。
如果它错了，解密值会差**一个 2 的整数次幂**——2 倍、4 倍、或者 2^某个荒谬的数。
而你测到的是 `2.0 → 2.004673`，差 **0.23%**。**所以 `corFactor` 是对的**，
这个方向不用查。

**但那行代码本身是个隐患，值得顺手打一行确认**：

```cpp
uint32_t correction = cc.GetBootPrecomputation(slots).correctionFactor - deg;   // Bootstrap.cu:204
```

`correctionFactor` 在 `BootstrapPrecomputation.cuh:36` 是 `uint32_t`。
如果它真的是 0 而 `deg > 0`，这个减法在无符号下**回绕成一个巨大的数**，
然后 `1ULL << llround(correction)` 是 **UB**（x86/PTX 会把移位数按 6 bit 取模，
结果是什么都可能）。

现在没炸，说明 `correctionFactor` **不是** 0——我们传给 `EvalBootstrapSetup` 的 0
是 OpenFHE 的"自动选择"约定，存进 precomputation 的是它算出来的值（sparse 通常是 9）。
打印 `correctionFactor` 和 `deg` 两个数确认一下即可，一行 `PRINT`。

---

## 4. 【要查】CtS/StC 里唯一的 FIXEDMANUAL 专用代码是我们自己加的

这条我必须先说，因为巧合太整齐：

* 精度问题在 **CtS/StC**；
* 只在 **FIXEDMANUAL** 下测到；
* 而 `CoeffsToSlots.cu` 相对上游 `fa97286` 的改动**只有一处**，
  并且它 **只在 FIXEDMANUAL 下生效**：`acb256f` 的 `alignToDiagonals`。

```cpp
const bool alignNeeded = cc.rescaleTechnique == FIXEDMANUAL;
const auto alignToDiagonals = [&alignNeeded](Ciphertext& ct, const LTstep& step) {
    if (!alignNeeded) return;
    int ptLevel = -1;
    for (const Plaintext& pt : step.A) {              // <- 取这一步所有对角线的
        const int level = pt.c0.getLevel();
        if (level >= 0 && (ptLevel < 0 || level < ptLevel))
            ptLevel = level;                          //    **最小** level
    }
    if (ptLevel >= 0 && ct.getLevel() > ptLevel)
        ct.dropToLevel(ptLevel);
};
```

它是为了修一个越界读（"would read 35 limbs from pt[0], which holds 34"）加的，
`dropToLevel` 本身对值是精确的，所以它**不应该**掉精度。但有一个假设它没有检查：

**它取的是一步之内所有对角线 level 的最小值，假设它们一样。**
如果同一步里的对角线在**不同** level 上，那么把密文降到最低的那个之后，
和更高 level 的对角线相乘就是在错配的 level 上做的——`multPt` 会按密文的 limb 数
去截取明文，而被截掉的那些 limb 携带的是这个明文在别的素数下的残数。
**值不会崩，但会错**——错的量级取决于差了几个 limb，而且**逐槽不同**，
因为不同对角线对应不同的槽。

**这和第 2 节观察到的"逐槽差 15 倍"是一致的。**

### 请打印这个

在 `alignToDiagonals` 里加一行，每步打印：

```
step i: ct.getLevel() = L,  diagonal levels = [min .. max],  distinct = n
```

* **`min == max`（所有对角线同 level）** → 我这个假设成立，这处清白，往别处查；
* **`min != max`** → 就是它。修法是按每个对角线各自的 level 对齐，
  或者把密文降到 **max** 而不是 min，再对低 level 的对角线单独处理。

这是一行打印就能二选一的事，而且它是我引入的，所以请优先查它。
如果确实是它，那么它同时解释了为什么**只有 FIXEDMANUAL 有这个精度问题**。

---

## 5. 你对 fmb=58 那个 NaN 的解释方向反了

> "q0/Delta=256 时消息值 2.0, 3.0 可能已经超出近似范围"

`q0/Delta` **变大**意味着消息**更靠近零**（`m/q0 = v / (q0/Delta)`，
v=2 时从 1/16 降到 1/128），正弦近似只会**更好**，不会更差。
所以 NaN 不是消息越界。

你给的第二个解释更可能：`depth=37` + `first_mod_bits=58` + `scaling_bits=50`
这个组合在素数选择或模链上出了问题。
值得看一眼构造引擎时有没有 warning，以及 `cc.prime[0].p` 实际是多少。
不过这是支线，不必为它停下。

---

## 6. 两个便宜的定位实验

如果第 4 节的打印显示对角线 level 一致（我那处清白），下面两个继续切：

### 6.1 bootstrap 一个**全零**密文

```python
ct = engine.encrypt(np.zeros(engine.slots, dtype=complex))
out = engine.bootstrap(ct)
err = np.abs(np.real(engine.decrypt(out)))
print(err.max(), np.quantile(err, [0.5, 0.99]))
```

零消息没有值可以让误差"成比例"，所以**测到的就是纯加性底噪**。

* 底噪 ~0.02 → 误差与消息完全无关，是某处在**加**东西；
* 底噪 ~1e-13 → 误差是消息驱动的，前面"不随值变化"的观察需要用更多样本重新审视
  （三个点太少）。

### 6.2 填满所有槽，看误差的**结构**

现在的消息只有 3 个槽非零、32765 个是零。CtS/StC 是跨槽的变换，
所以少数几个非零槽的误差会被摊到所有槽上，只看 3 个槽看不出形状。

填一个已知图案（比如 `0.5 * cos(2*pi*k*i/slots)`，或者干脆全 1.0），
报告误差的 **p50 / p99 / max**，外加它和槽号的关系——
比如按 `slot % 16`、`% 128`、`% 2048` 分组各自的均值。

* 误差**均匀无结构** → 是噪声，往 key-switch / rescale 查；
* 误差**按某个周期分组** → 是变换里某一步错了，那个周期直接指出是哪一层。

---

## 7. 测试那处改动 —— 接受

`_float_scalar_on_degree_two` 先 rescale 再乘，对。
它要测的"`multScalar` 必须缩放 c2"这件事没丢，degree 仍然是 2，只是 scale 回到了 Delta^1。
