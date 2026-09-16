# 我那个判据写错了。相关系数 0 证明不了"随机噪声"——它指向放大

日期：2026-09-16。针对 `fc19d66`。

---

## 1. 先认错：实验 1 的判据是我写坏的

我上一封给的代码是：

```python
first  = decrypt(bootstrap(encrypt(zero)))
second = decrypt(bootstrap(encrypt(zero)))
```

**`encrypt` 调了两次，所以这是两个不同的密文**——各自带不同的加密噪声 `e`。
两者的 bootstrap 结果不相关，只说明**输出依赖输入的噪声**，
说明不了"bootstrap 自己注入了随机性"。我的判据表述错了，你按它跑没有问题，
是我给的判据不成立。

### 而且"bootstrap 注入随机噪声"在机制上不可能

CKKS 的**求值过程没有任何随机数来源**：key switch 用的是固定的钥匙，
NTT、rescale、旋转、明文乘全是确定性算术。随机性只在 `encrypt` 和 keygen 里。

所以 corr ≈ 0 只剩一个解释：**bootstrap 把输入自带的加密噪声放大了。**

---

## 2. 放大倍数是多少

一次新鲜加密的噪声，在消息域大约是：

```
sigma_e ~ 3.2 (每个系数)，正则嵌入下 ~ sigma_e * sqrt(N) = 3.2 * 256 = 819
换算到消息域：819 / Delta = 819 / 2^50 ~ 7.3e-13
```

而 bootstrap 输出的 σ 是 **0.0154**。

**增益 ≈ 0.0154 / 7.3e-13 ≈ 2.1e10 ≈ 2^34。**

一次 key switch 只放大到 4.23e-10（你实验 2 测的），也就是约 2^9。
**bootstrap 的 2^34 比它多 25 个 bit**，这不是"累积"能解释的——
你自己也算过，4000 次旋转累积才 2e-6。

**所以要解释的是这个 2^34，不是"噪声从哪来"。**

---

## 3. 两个修正后的实验

### 3.1 确定性检查（修正版）

`encrypt` **只调一次**，对**同一个密文**跑两次 bootstrap：

```python
ct = engine.encrypt(np.zeros(engine.slots, dtype=complex))
first  = np.real(np.asarray(engine.decrypt(engine.bootstrap(ct))))
second = np.real(np.asarray(engine.decrypt(engine.bootstrap(ct))))
print("差:", np.abs(first - second).max(), " vs 各自:", np.abs(first).max())
```

**预期：完全一样（差 < 1e-9）。** 如果不一样，那是另一类问题——
求值里不该有随机性，出现了就说明有人在读未初始化的内存。

### 3.2 增益测量（这才是要紧的那个）

拿两个**同值、不同噪声**的输入，看输出噪声跟不跟：

```python
fresh = engine.encrypt(zero)
noisy = engine.encrypt(zero)
for _ in range(1000):
    noisy = engine.rotate(noisy, 1)        # 每次约 4e-10，累积到 1e-8 ~ 1e-6
```

分别测 bootstrap 前后的幅度，报四个数：`fresh` 前/后、`noisy` 前/后。

* **输出噪声跟着输入涨** → bootstrap 是放大器，增益就是要解释的量；
* **两个都是 0.015** → 它有自己的底噪，和输入无关，那增益那条线作废，
  要改查"底噪从哪来"。

**这两条已经写成测试推上去了**（`test_bootstrap_noise_level.py` 的
`test_bootstrapping_one_ciphertext_twice_gives_the_same_answer` 和
`test_bootstrap_noise_scales_with_the_input_noise`），后者是打印四个数、不设阈值——
因为是哪一种还没定，设一个猜的阈值只会把猜测固化进去。

---

## 4. 如果确实是放大，2^34 能从哪来

只列方向，不下结论——等 3.2 的数。

**最像的一条是中间缩放。** `Bootstrap.cu:250`：

```cpp
double constantEvalMult = pre * (1.0 / (k * cc.N));
ctxt.multScalar(constantEvalMult, false);
```

`cc.N = 65536`，所以这里把密文**缩小了 2^16 以上**（还要乘 `pre = 2^-deg`）。
Chebyshev 和 double-angle 在这个缩小后的尺度上做，最后靠
`multIntScalar(ctxt, corFactor)`（`:318`）放大回去。

**在缩小的那一段里，每次 rescale 的舍入误差在消息域是固定的 ~1/Delta；
等最后放大回去时，这个误差跟着被放大同样的倍数。** 这正是"绝对误差、
与消息无关、与 `q0/Delta` 无关"的形状——和你前面测到的三条特征全部吻合。

要确认它，打三个数就够：

1. `pre`、`k`、`constantEvalMult` 的实际值（`Bootstrap.cu:250` 附近，PRINT 块已经有一行打 `constantEvalMult`）；
2. `correction` 和 `corFactor`（`:318`），看放大倍数是多少；
3. 在 `multScalar(constantEvalMult)` 之后、`multIntScalar(corFactor)` 之前，
   各解密一次看幅度——**缩小了多少倍，就是误差被放大多少倍的上限**。

如果 `1/(k*N)` 是 2^-20 量级而 `corFactor` 把它乘回来，
那 2^34 里就有 20 个 bit 是这么来的，剩下的 14 个 bit 再找。

---

## 5. 你的排除表我全部同意，补一行

| 假设 | 状态 |
|---|---|
| alignToDiagonals 对角线 level 不一致 | ❌ 排除 |
| 误差是乘性/消息驱动 | ❌ 排除 |
| 周期 2048 结构 | ❌ 排除（统计假象） |
| key switch 本身有噪声 | ❌ 排除（单次 4e-10） |
| ~~确定性误差（常数表/明文编错）~~ | **⚠️ 未排除** —— 判据是我写坏的，见 §1；用 3.1 重测 |
| **bootstrap 放大输入噪声约 2^34** | **← 现在的主假设，用 3.2 测** |

---

## 6. 顺带

`test_a_single_rotation_is_accurate` 测到 4.23e-10 通过了，很好——
这条留在套件里，因为它同时是 key switch 的健康线。
