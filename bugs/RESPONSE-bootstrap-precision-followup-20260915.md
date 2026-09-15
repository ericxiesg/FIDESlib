# 三个数收到。两条结论要改，一条悬案就此结掉

日期：2026-09-15。针对 `d092331`。

---

## 1. 三个数都很有用，先谢

* **10.9 bit** —— 落在我模型里 9.3–12 bit 那一档，正是重现 1e178 的区间；
* **`07c.denominator` p50 = 7.03e-05 = 0.14 × inv_epsilon** —— 越界假设成立；
* **`07d` p50 = 1.9e+159** —— 发散在**数据槽**，不只是填充槽。分位数探针立刻起了作用。

---

## 2. 【要改】NoiseLevel=3 那条推断，位置找错了

你的推断是：

> `approxModReduction` 末尾的 rescale 把 NoiseLevel 减 1 ⟹ 进入前是 3

前提是 `approxModReduction` 之后没有别的东西动 NoiseLevel。**但有，而且有三处。**
`src/CKKS/Bootstrap.cu`，`approxModReduction` 在 :291，往后：

```cpp
if (ctxt.NoiseLevel == 2) { ctxt.rescale(); }      // :300-302  ← 已经把它归一到 1 了
...
EvalCoeffsToSlots(ctxt, slots, true);               // :309      最后一次 StC
...
multIntScalar(ctxt, corFactor);                     // :318      整数乘，不改 NoiseLevel
return;                                             //           之后没有 rescale
```

**`:300` 这一行已经把 `approxModReduction` 的输出归一了。** 所以无论它留下 1 还是 2，
到 `:309` 之前都是 1。而输出是 2，这个 2 只能是 `EvalCoeffsToSlots` 留下的。

再往下一层，`CoeffsToSlots.cu` 和 `LinearTransform.cu` 的约定是**入口归一、出口不归一**：

* `EvalCoeffsToSlots` 开头 `:96-97` `if (ctxt.NoiseLevel == 2) ctxt.rescale();`
* `LinearTransform` 开头 `:74`、`:266`、`:461` 同样
* 两者**结尾都没有**对应的 rescale

而 StC 是一串明文乘，每次 `NoiseLevel += 1`。所以 `EvalCoeffsToSlots` 必然在 2 上结束，
`multIntScalar` 不改它，**输出必然是 2**。这是从控制流直接读出来的，不用跑。

### 后果一：它**不是**精度问题的另一面

漏掉的那次 rescale 在**整个 bootstrap 的最后**，在 StC 之后。
那时候值确实在 Δ²，metadata 也说 Δ²——**两者是一致的，没有信息丢失**。
`Engine.bootstrap` 补一次 rescale，花掉一格 level，**精度分毫不损**
（你自己的数据也支持：解密值 2.0 → 2.004673，如果 metadata 和值不一致，偏差会是 Δ 倍，不是 0.2%）。

**所以"同一个 bug 的两面"不成立，是两个独立的问题。**
你建议的下一步"在 `evalChebyshevSeries` 和 `applyDoubleAngleIterations` 里加 PRINT
找 NoiseLevel 从 1 变 3 的那一步"——那里找不到，因为 `:300` 在它们之后。

### 后果二：真正的修法位置

按 FIDESlib 自己的约定（**被调方在入口归一**），`Bootstrap()` 返回 NoiseLevel 2
其实是**符合约定**的：下一个消费者本来就会在入口归一。
问题出在**边界**：FIXEDMANUAL 的 API 消费者（`EvalMult`、`EvalAdd`）**不归一**——
那正是 FIXEDMANUAL 的定义，scale 由调用方管。

所以正确的落点是 **`api/CryptoContext.cpp::EvalBootstrap`**，不是 Python 包装层，
理由写清楚："在边界归一，因为 FIDESlib 内部用的是被调方归一的约定，而 FIXEDMANUAL 的调用方不用"。
我先前那条 Python 侧的 rescale 可以搬下去，行为不变。

---

## 3. 【结案】depth 的账是对的，不用上 38

我先前警告过：这条 rescale 多花一格，`resolve_bootstrap_depth` 取的 17 是加它之前测的，
所以可能要上 depth 38。**你的实测把这件事结掉了，结论是不用改**：

```
Fresh:              level=37
Raw EvalBootstrap:  level=21     -> 本体吃 16 格
After rescale:      level=20     -> 合计 17 格
```

而 `resolve_bootstrap_depth` 返回 **17**，`achievable = 37 − 17 = 20`——**和实测的 20 精确吻合**。

（表里那个 17 的注释写的理由是"GetBootstrapDepth 报 16 + `EvalCoeffsToSlots` 对齐对角线多花 1"，
和现在这个"本体 16 + 边界 rescale 1"不是同一个说法。**数对了，理由待查**——
但这不影响使用，先记一笔。）

**所以 depth 37 稳，`MEASURED_BOOTSTRAP` 不用动，我先前那条 depth 38 的警告作废。**

---

## 4. 【要改】`clear.py` 的 float scalar 守卫，我回退了

你的改动和注释：

> "float scalar (multScalar) does NOT assert and is legal on degree-2, so we test the plaintext path."

两个问题：

**(a) 把 degree 和 scale 混了。** 我那条守卫 `_require_canonical_scale` 查的是 `scale_exp`
（也就是 NoiseLevel），不是 `degree`。"`multScalarNoPrecheck` 会缩放 c2"说的是 **degree**
——那个确实没问题，但和守卫无关。

**(b) "does NOT assert" 与代码不符。** `EvalMult(ct, double)` 走的是
`res_gpu->multScalar(scalar)`（`api/CryptoContext.cpp:1298`），即
`Ciphertext::multScalar(const double, bool)`（`Ciphertext.cpp:902`）：

```cpp
	if (cc.rescaleTechnique == FLEXIBLEAUTO || ... || FIXEDAUTO) {
		...
		if (NoiseLevel == 2) this->rescale();
	}
	assert(this->NoiseLevel == 1);            // <- :913，在 if 外面，FIXEDMANUAL 也走到
	multScalarNoPrecheck(c, rescale && cc.rescaleTechnique == FIXEDMANUAL);
```

而 `multScalarNoPrecheck` 里 `NoiseLevel += 1` 并且
`NoiseFactor *= ScalingFactorReal.at(...)`——**那是 degree-1 的缩放因子**。
在一个 Δ² 的操作数上，metadata 出来描述的是一个它没有的 scale。

**已回退**，并把上面这段证据写进了 `_require_canonical_scale` 的 docstring，免得再来一轮。

### 顺带：那次改动产生了一个重名函数

改完之后 `test_clear_engine_contract.py` 里有**两个** `test_plaintext_multiply_refuses_an_unrescaled_ciphertext`
（原来的第 44 行，和 float 那条被改名后的第 51 行）。Python 只保留后一个，
所以 **float scalar 那条用例被静默删掉了**，而 "207 passed" 看不出来。

这类事值得加一道机械检查。我这边跑的是：

```
grep -n "^def test_" <file> | awk -F'[ (]' '{print $2}' | sort | uniq -d
```

---

## 5. 10.9 bit 该往哪查

既然它和 NoiseLevel 那件事无关，得单独定位。**你自己的数据已经给了第一条线索**：

```
expected  = [2.0,      -3.0,      0.5     ]
abs_error = [4.67e-3,  1.11e-3,   1.70e-2 ]
```

**误差不随值缩放**——最小的那个值（0.5）反而误差最大。所以这是**绝对误差**，
由界 `q0/Delta = 32` 决定，不是相对误差。

这就把嫌疑分成两类：

| | 误差性质 | 嫌疑 |
|---|---|---|
| 模约简部分（ModRaise + Chebyshev + double-angle） | **绝对**，随 `q0/Delta` 缩放 | ✅ 与观测一致 |
| 线性变换部分（CtS / StC） | **相对**，随值缩放 | ❌ 与观测不一致 |

**一个参数就能证实**：把 `first_mod_bits` 从 55 改成 60（`q0/Delta` 从 32 变成 1024），
其余不动，重测同样三个值的绝对误差。

* 误差涨到 ~0.5（32 倍）→ 确定在模约简部分，往 Chebyshev / double-angle / correctionFactor 查；
* 误差基本不变 → 在线性变换部分，往 CtS/StC 查。

这比加 PRINT 便宜得多，而且结论是二选一的。

顺带两个可以一起看的：

* **`correctionFactor`**：我们给 `EvalBootstrapSetup` 传的是 **0**（`pyfideslib/__init__.py:103`）。
  thor-openfhe 的修正是 12 → 0，所以我们这一项已经对。但要确认 `Bootstrap.cu:204`
  的 `correction = correctionFactor - deg` 在 `correctionFactor = 0` 时是不是负数——
  如果是，`corFactor = 1 << llround(correction)` 会是个奇怪的值。
* **`MODRAISE_WITH_P0`**（`Bootstrap.cu:275`）：这个编译期开关会多一次 rescale，
  开或不开都值得记一笔。

---

## 6. 标定那一层同意，但顺序要注意

你说"即使标定完美，10.9 bit 的 bootstrap 在 2e-4 的分母上也会崩"——同意。
但反过来也成立：**即使 bootstrap 修到 22 bit，p50 = 7e-5 的分母仍然低于 `inv_epsilon` 7 倍**，
按我量的饱和表那是 50% 以上的误差。**两层都得修，缺一不可。**

顺序建议：先 bootstrap（它是根因，而且会改变第二层的表现），
修好之后再用真实激活重新量分母，那时候标定要调多少才有意义。

---

## 7. 其余改动

* `bench_params` 改回 `dnum=4` —— ✅ 谢，这样它才真的对应 bench
* 容差改回 `1e-3` 并标 `xfail` —— ✅ 正确，比放宽好
* `test_eval_bootstrap_itself_returns_a_canonical_ciphertext` 标 `xfail` —— ✅ 本来就是这个用途
* `conftest.py` 的 `SMALL` 我这边补了 `SPARSE_TERNARY`：之前**整个 GPU 套件都跑在
  `UNIFORM_TERNARY`**（Engine 默认），和 bench 的 sparse 不是同一套 Chebyshev 系数和
  double-angle 次数。详见 `report/part21-review-20260915.md` §3。
