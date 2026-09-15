# 回复：bootstrap 后 NoiseLevel=2（`BUG-bootstrap-noise-level-2-20260915.md`）

日期：2026-09-15。结论：**修法正确，合入了**（`72dc818`）。但它带来一笔没算的
level 账，以及一处会漏的情形；另外对剩下的 Goldschmidt 问题，我有两个你列的猜测
可以直接排除——不是靠推理，是靠读代码。

---

## 1. 修法是对的

判据是你自己给的那一行：

```
After bootstrap:  level=21 noise=2
After rescale:    level=20 noise=1     解密值 0.99787 → 0.99787
```

值没变，说明密文**确实在 Δ²**，不是标错了。如果只是标签错，rescale 会把值除掉一个 Δ。
所以这条 rescale 是必要的，不是绕过。

---

## 2. 但它多花一格 level，而 level 的账没跟着改 [需要你确认]

`bench.py:129` 是这么推 bootstrap 落点的：

```python
achievable = args.depth - resolve_bootstrap_depth(args)
```

`resolve_bootstrap_depth` 取 `budget.MEASURED_BOOTSTRAP[(16, 32768, (3,3))]["bootstrap_depth"] = 17`,
**那是加这条 rescale 之前测的**。现在 `Engine.bootstrap` 比 `EvalBootstrap` 多花一格，
所以这个数在 FIXEDMANUAL 下应该是 **18**。

`bench.py:126-131` 自己的注释写着这件事的后果：

> Planning against a different number silently builds keys for levels the run never reaches -
> and, worse, hides that the level budget does not fit at all.

具体影响：两次 bootstrap 之间最深的一段（softmax 之后到 attention dense）要 **19 格**，
约束是 `depth - bootstrap_depth ≥ 19`：

| `Engine.bootstrap` 吃几格 | 最小 depth | depth=37 |
|---:|---:|---|
| 17（旧，无 rescale） | 36 | 留 1 格 |
| 17（`EvalBootstrap` 16 + rescale 1） | 37 | **零余量** |
| 18（`EvalBootstrap` 17 + rescale 1） | **38** | **跑不完** |

**是 37 零余量还是 38，完全取决于第 2 节末尾那个 16 / 17 的问题**——这是现在最要紧的一个数。

**[实测]** 在 `ClearEngine` 上把 bootstrap 落点从 20 降到 19，整层直接失败：

```
depth  bootstrap落点  结果
   37            20  OK   出口 level 1
   37            19  FAIL rescale: would leave the ciphertext at level -1
   38            20  OK   出口 level 1
```

所以**你报告里"depth budget of 37 has enough headroom"这句无论如何都不成立**：
最好的情况是零余量，最坏是要上 depth 38。

你那次跑之所以没在这里炸，是因为它只跑了 layer 0 的前半段就被 softmax 拦住了
（`inverse_denominator` 出 1e178），**还没走到最深的那一段**
（softmax 之后一路到 attention dense，要 19 格）。等 Goldschmidt 修好，
这个 level 墙就会立刻撞上——所以这两件事得一起解决，不能只修一个。

### 这笔账和旋转钥匙是耦合的 [实测]

depth 38 每把钥匙多一层 limb，直接吃掉 0.44 GiB 余量。旋转次数与 depth 无关，
所以代价全在内存上：

| depth | 额外 key | 总 key | 旋转/层 | 旋转钥匙 | 总驻留 | 32 GB 余量 |
|---:|---:|---:|---:|---:|---:|---:|
| 37 | 0 | 15 | 8138 | 2.52 G | 28.34 G | 1.46 G |
| 37 | 4 | 19 | 4413 | 3.15 G | 28.97 G | 0.83 G |
| 37 | 6 | 21 | 4117 | 3.46 G | 29.28 G | 0.52 G |
| 38 | 0 | 15 | 8138 | 2.58 G | 28.78 G | 1.02 G |
| 38 | 4 | 19 | 4413 | 3.22 G | 29.42 G | **0.38 G** |
| 38 | 6 | 21 | 4117 | 3.54 G | 29.74 G | **0.06 G** |

**所以这条 rescale 如果逼到 depth 38，旋转钥匙的预算就从 +9 把缩到 +4 把**，
而且 0.38 GiB 已经很薄。这就是为什么第 2 节末尾那个 16 / 17 的数要先测出来：
它同时决定 depth 和我们能买得起几把钥匙。

两个数我这边还确认不了，要你在卡上测：

* **`EvalBootstrap` 在 FIXEDMANUAL 下到底吃几格？** 你的复现打的是 37→21 = **16**，
  而我们表里记的"有效值"是 17（OpenFHE 的 `GetBootstrapDepth()` 报 16，
  `EvalCoeffsToSlots` 对齐对角线再吃一格）。你那次复现只建了 6 把旋转钥匙、dnum=3，
  和 bench 的配置不同，所以可能不是同一个数。**请在真实 bench 配置下打印
  bootstrap 前后的 level**，我好把 `MEASURED_BOOTSTRAP` 改对。
* 改对之后，`resolve_bootstrap_depth` 要在 FIXEDMANUAL 下 +1，我来改。

---

## 3. 一个会漏的情形：输入不在 NoiseLevel 1 时

现在是无条件 rescale 一次：

```python
out = self.cc.EvalBootstrap(x)
if self.scaling_technique == _core.FIXEDMANUAL:
    out = self.cc.Rescale(out)
```

如果 `EvalBootstrap` 的输出 NoiseLevel 跟输入有关（输入是 2 时输出 3），
rescale 一次就只到 2，问题原样复现，而且这次**没有那条报错**能把它抓出来。
`he_inv:182` 的 `self.bootstrap(denominator)` 收到的 denominator 来自
`_sum_over_groups(exp_u)`，上游有 `rescale`，应该是 1——但"应该"正是这个项目一直吃亏的词。

建议改成收敛到 1，并且先要求输入就在 1：

```python
if self.scaling_technique == _core.FIXEDMANUAL:
    before = self.noise_level(x)
    if before != 1:
        raise ValueError(
            f"bootstrap wants a canonical ciphertext, got noise level {before}. Rescale first: "
            f"how many levels the bootstrap costs depends on it, and the level plan assumes one.")
    while self.noise_level(out) > 1:
        out = self.cc.Rescale(out)
```

`while` 而不是 `if`，是因为多花的每一格都必须被上面那本账看见；
入口的断言是为了让"花几格"成为一个定数，否则 level plan 就没法算。

---

## 4. Goldschmidt 的 1e178：你列的两个猜测可以排除

你写的是：

> `_restore_magnitude` (line 208) does `add(ct, conjugate(ct))` and `multiply(ct, factor)`
> which may not correctly track NoiseLevel under FIXEDMANUAL.

这两条我查了代码，都不是：

* **`multiply(ct, factor)`**：`factor` 是 `max(int(...), 1)`，Python `int`。
  `pyfideslib/__init__.py:156` 对 int 走 `EvalMultByInteger`——级和尺度都不动。
  `ClearEngine.multiply`（`clear.py:146`）对 int 的建模完全一致。两边不存在分歧。
* **`add(ct, conjugate(ct))`**：`conjugate` 现在头上有 `requireDegreeOne`
  （`Ciphertext.cpp`），degree-2 会直接抛。没抛，说明 degree 是对的。

顺带排除第三个更像的嫌疑：`he_inv` 的修正项 `2/k*delta - b` 走的是
**浮点标量减密文**，而 `EvalScalarSub` 在 GPU 路径上是 `negate(); addScalar(scalar)`
= `scalar - ct`（`api/CryptoContext.cpp:1105-1128`，注释里就写着 "this is the path
he_inv takes"）。符号是对的——符号反了 Goldschmidt 一定发散，本来是最像的解释。

**所以我不再猜。** 这个 bug 我已经在别的形态上连猜错三次方向，每次都是实测纠回来的。

---

## 5. 要两样东西来定位

### 5.1 先分清是数据槽炸了还是填充槽炸了

`07d` 报的 `max +2.926e+178` 是**全槽最大值**。`he_inv` 的 `ones` 只把
*数据槽*约束住；填充槽在明文模型里精确为 0，在设备上是 ~1e-8 的加密噪声，
Goldschmidt 在那里没有收敛目标。所以 1e178 完全可能只出在填充槽，
而数据槽是好的——那样的话 bug 在别处，`he_inv` 本身没问题。

现在的探针分不出来：它用"非零槽"当数据槽，而设备上**没有槽是精确零**。
我把它改成报 `|x|` 的分位数（`bench.py:371`）：

```
|x| p50 ...  p99 ...  max ...
```

* 数据槽发散 → p50 跟着涨；
* 只有填充槽发散 → p50 正常、max 爆炸。

**请拉最新的 bootstrap-dev 重跑一次带 `--probe` 的，把 07b/07c/07d 三行贴回来。**

### 5.2 he_inv 逐次迭代的 (level, noise_level)

`ClearEngine` 精确记账 level 和 scale，我可以给出**应该**是什么；
你在设备上打同样的序列，第一处对不上的地方就是 bug 的位置。
这正是之前几次定位管用的方法。

请在 `he_inv` 的循环里（`numeric.py:186-198`）每次迭代打印：

```
iter N  a: level=.. noise=..   b: level=.. noise=..   |a| p50=..  |b| p50=..
```

期望序列在下面。

### 5.3 `ClearEngine` 给出的期望序列 [实测]

depth 37、bootstrap 落 20、layer 0。`he_inv` 一层里调两次，**下面是 softmax 那次**
（`epsilon = 0.00049 ≈ 2^-11`，和 `softmax1` 的 `inv_epsilon` 对得上）：

```
   step | a.lvl a.nz | b.lvl b.nz |    a.delta    b.delta |    |a| p50    |a| max |    |b| p50    |b| max | error
   init |    20    1 |    20    1 |  1.000e+00  1.000e+00 |  1.000e+00  1.000e+00 |  2.166e-04  2.166e-04 | 0.00049
 iter 1 |    19    1 |    19    1 |  2.502e-01  2.502e-01 |  1.000e+00  1.000e+00 |  2.167e-04  2.167e-04 | 0.00195
 iter 2 |    18    1 |    18    1 |  1.572e-02  1.572e-02 |  2.506e-01  2.506e-01 |  5.428e-05  5.428e-05 | 0.00777
 iter 3 |    17    1 |    17    1 |  3.888e-03  3.888e-03 |  2.452e-01  2.452e-01 |  5.312e-05  5.312e-05 | 0.03062
 iter 4 |    16    1 |    16    1 |  3.903e-03  3.903e-03 |  9.426e-01  9.426e-01 |  2.042e-04  2.042e-04 | 0.11531
 iter 5 |    15    1 |    15    1 |  3.903e-03  3.903e-03 |  3.222e+00  3.222e+00 |  6.979e-04  6.979e-04 | 0.37080
 iter 6 |    14    1 |    14    1 |  3.893e-03  3.893e-03 |  8.155e+00  8.155e+00 |  1.766e-03  1.766e-03 | 0.78932
 iter 7 |    13    1 |    13    1 |  3.906e-03  3.906e-03 |  1.365e+01  1.365e+01 |  2.957e-03  2.957e-03 | 0.98614
 iter 8 |    12    1 |    12    1 |  3.881e-03  3.881e-03 |  1.691e+01  1.691e+01 |  3.662e-03  3.662e-03 | 0.99995
```

怎么用这张表：

1. **`a.nz` / `b.nz` 必须全程是 1。** 这是最强的一条检查，也是最可能出问题的一条——
   bootstrap 那个 bug 就是 NoiseLevel 从 2 起步。设备上任何一步出现 2，那里就是现场。
2. **level 每次迭代掉 1，20 → 12，一共 8 次迭代。** 多掉或少掉都说明某个原语
   在设备上花的级数和 Python 以为的不一样。
3. **`b / b.delta` 必须收敛到 1**，这是 Goldschmidt 的不变量：
   末行 `3.662e-03 / 3.881e-03 = 0.943`，误差列同时到 0.99995。
   设备上这个比值如果开始远离 1，迭代就在发散，**从哪一次迭代开始偏就是答案**。
4. **`|a|` 的包络是 1.0 → 16.9，全程不超过 17。** 出现 1e178 意味着偏了 176 个数量级，
   不是精度问题，是结构问题。

一个说明：这次 dry run 的权重全是 0，所以所有槽的值相同，`p50 == max`。
**幅值两列只能当包络看，不能当分布看**；真实权重下 5.1 节那个 p50/max 分裂才有意义。
level / noise / delta 三列是精确记账、与输入无关，那三列可以逐格对。

---

## 6. 顺带：这轮我推的东西

`7757939`：旋转基改成实测挑选。`--extra-rotation-keys 4` 让一层的旋转次数
从 8138 降到 4413（−46%），钥匙从 15 把加到 19 把，预测驻留 28.97 GiB、
32 GB 卡余 0.83 GiB。**这个余量是 `budget.estimate` 的预测，麻烦用
`--device-memory` 核一下真实驻留**，对得上我就把它设成默认。

细节见 `report/optimisation-plan-batch1-20260915.md`。
