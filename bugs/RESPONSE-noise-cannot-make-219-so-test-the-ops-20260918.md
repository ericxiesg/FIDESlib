# ClearEngine 造不出 219.6：`b` 不是更吵，是不一样

日期：2026-09-18。承接 `4475c84`。

---

## 0. 结论

你定位到 `iter03_b` 是对的。我在 ClearEngine 上**试了三条放大机制，注入噪声，都造不出 219.6**，
而且把迭代里每一个操作数都量了，全都是良性的。

**所以设备上的 `b` 不是一个更吵的值，是一个不同的值。** 这不是精度问题，是某个算子算错了。

---

## 1. 先把 `a22ec04` 摘出去

你说旧代码 15-key 干净、新代码 15-key 炸，怀疑 `a22ec04` 动了 `stages.py`(4 行) / `dense.py`(1 行)。
查过了：

```
stages.py  4 行：  "a layer's 8138" -> "8258"     （注释）
clear.py   2 行：  同一处注释
dense.py   1 行：  + self.probed("10.attention_dense", out)
```

**没有一行改计算。** 那个 commit 里唯一能影响设备行为的是给 `plan_rotations` 传了 `compact`。
我把两种计划的 key level 逐把对比了（`--binary-rotations`，depth 37）：

```
key      compact=F  compact=T
1..16384    30/31      30/31     ← 15 把全部相同
keys the compact plan truncates further: none
```

**完全一致。** `a22ec04` 对 15-key 设备跑是空操作。

（另外提一句：你说的"旧代码干净"那次，整体 `hidden MAE` 是 2.402e+19，
也就是说那次**同样是坏的**，只是坏在别处。所以"旧干净/新炸"可能不是代码差异，
而是两次跑的发散位置不同。）

---

## 2. 三条放大机制，全部量过

真实几何、`epsilon=2^-11`、独立注入泄漏：

| 注入位置 | 结果 |
|---|---|
| **`ones` 的空槽** | 输出空槽 = 泄漏 × **1.158e4**，跨 10 个数量级严格线性，**不饱和** |
| **分母的空槽** | **没有影响**（1e-2 的泄漏反而让输出空槽变成 1e-10） |
| **承载槽上的相对噪声** | **不放大**：1e-3 进 → 3.1e-3 出，线性透传 |

第一条修正了我 `ad46a10` 里"饱和在 1"的说法——那是因为我当时**两边同时泄漏**，
b 的泄漏把比值钉住了。单独漏 `ones` 是线性无界的，**你 §3 说的不对称是对的**。

但它对不上你的数据：

- 放大只出现在 **`a`** 上，`b` 全程 0.0039 不动（我逐轮打了，见下）。你的设备是 **`b` 先炸**。
- 要让 `a` 到那个量级需要 `ones` 空槽泄漏 ~1e-2；而你 `07c.denominator` 的 p50 是 1.012e-11。

我这边 `leak(ones)=1e-2` 的逐轮，注意 b 那一列：

```
iter01_a max 0.9505  | iter01_b max 0.2502
iter02_a max 0.1931  | iter02_b max 0.01572
iter03_a max 0.07402 | iter03_b max 0.003888   <- 你的设备这里是 219.6
iter05_a max 0.06726 | iter05_b max 0.003903
iter08_a max 0.4496  | iter08_b max 0.003881
```

**`b` 在 ClearEngine 上无论怎么注入噪声都不动。**

---

## 3. 迭代里每个操作数都量了，没有一个危险

我怀疑过 iter03 的 `correction = 2/k·δ_b − b` 在那一轮量级相撞（0.0158 对 0.0164），
以及 `_restore_magnitude` 的整数缩放溢出。都量了：

```
EvalScalarSub 的标量 2/k·b.delta：1.00, 0.251, 0.0158, 0.00401, 0.00435, 0.00535, 0.00697
                                              ^^^^^^ iter03，最小，但仍是 O(1)
EvalMultByInteger 的整数：        31, 486, 412, 272, 161, 129      最大 2^8.9
```

标量 ≤ 1.0，整数 ≤ 486。**没有溢出的余地**，也没有大数相减。
而且 §2 第三行已经证明：即使在那个相撞点上注入噪声，迭代也不放大。

---

## 4. 所以请直接测算子

迭代里一共这几个算子，其中**只有一个从来没有在设备上单独测过**：

| 算子 | 调用形式 | 设备上测过吗 |
|---|---|---|
| **`EvalScalarSub`** | `subtract(2/k*δ_b, ct)` | **没有** ← 唯一空白 |
| `EvalMultNoRelin` + `EvalRelinearize` + `Rescale` | `_times` | 每个 stage 都在用 |
| `EvalMultByInteger` | `multiply(ct, factor)` | 已确认两边模型一致 |
| `EvalConjugate` + `EvalAdd` | `add(ct, conjugate(ct))` | stage 10 也在用 |

`EvalScalarSub` 我盯了好几天，一直只是"读代码看着是对的"。现在它是迭代里唯一没被排除的东西，
而且 `b` 的更新完全经过它（`b ← b·correction`，`correction` 就是它的输出）。

**请跑这个（两分钟，不用跑整层）：**

```python
import numpy as np, pyfideslib
e = pyfideslib.Engine("cuda:0", log_n=16, depth=37, scaling_bits=59, first_mod_bits=60, dnum=4)
x = np.linspace(-0.5, 0.5, e.slots)
ct = e.encrypt(x)
for s in (1.00049, 0.250732, 0.0158389, 0.00400755):   # he_inv 真实用到的标量
    for drop in (0, 10, 20, 24):                       # 不同 level
        c = e.level_down(ct, by=drop) if drop else ct
        got = np.real(e.decrypt(e.subtract(s, c)))
        want = s - x
        print(f"s={s:<10g} level={e.level(c):<3} max abs err {np.abs(got-want).max():.3e} "
              f"max |got| {np.abs(got).max():.4g}")
```

`want` 是 `s - x`，全程 O(1)。**任何一行的误差不是 1e-10 量级，就找到了。**
顺带把 `e.multiply(ct, 486)` 和 `e.add(ct, e.conjugate(ct))` 也这样对一遍，三个算子一起清掉。

如果三个都干净，那问题在 `prepare_for_multiply` 或者在这些算子的**组合/别名**上
（`correction` 在一轮里被用两次，先 `a` 后 `b`——如果第一次乘法就地改掉了它，
第二次就会拿到坏的，而这正好是"`b` 比 `a` 先坏"的形状）。那种情况请告诉我，我去读 `multNoRelin`。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

---

## 5. 附：那个测试已经写成 pytest 了，不用手抄

`python/tests/test_he_inv_primitives.py`（本次一并推）。用现成的 `engine` fixture，
所以 `cpu` 和 `cuda:0` 各跑一遍——**如果 cpu 过而 cuda:0 不过，就定位到设备算子了**。

```
cd python && PYFIDESLIB_DEVICES=cpu,cuda:0 python -m pytest tests/test_he_inv_primitives.py -q
```

49 个用例：

| 测什么 | 参数 |
|---|---|
| `subtract(标量, 密文)` | `he_inv` 真实用到的 5 个标量 × 4 个 level |
| `multiply(密文, 整数)` | 真实的 6 个因子 × 4 个 level，并断言 **level 不变** |
| `add(ct, conjugate(ct))` | 4 个 level，断言 level 不变 |
| **一整步 Goldschmidt，连做 5 步** | 断言**量级不增长**（不是断言精度——`want` 最后衰减到 1e-9，在噪声底上断精度是脆的） |

最后一个是重点：单个算子都在容差内、而组合起来发散，是完全可能的，
而设备上看到的正是"一步之内 0.0039 → 219.6 然后每轮平方"。

我这边没有 CUDA，跑不了，但我用一个精确算术的假引擎把 49 个用例的**函数体**都执行过了，
不会因为拼错 API 浪费你一轮。
