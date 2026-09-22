# 那是**第一次** `he_inv`，不是第二次：`0.3632 × 3 = 1.0897`，设备的 lift 天花板是 2

日期：2026-09-22。回应 `20032f8`。**收获很大，两条线一条关死、一条定案。**

---

## 0. 结论

| | |
|---|---|
| `he_exp` 窗口 | ✅ **关死**。38.1 在被 mask 清零的槽里，存活槽的 max 是 10.203 |
| 日志里的两次 `he_inv` | ❌ **是同一次**。`[range]` 打印和 `ValueError` 出自同一个调用 |
| `1.0897` 哪来的 | **`07c` max 0.3632 × `--inverse-lift 3` = 1.0896**。精确对上 |
| 该怎么办 | **`--inverse-lift 2`**。设备天花板是 `1/0.3632 = 2.75`，3 超了 |
| 我上一份说"别试 lift 2" | ❌ **撤回**，见 §3 |

---

## 1. `he_exp` 窗口：关死了

你的 §1 在纠结"07a carried max 38.1，但 he_exp observed max 10.203，对不上"。
**对得上，两个用的不是同一张 mask。**

- `07a` 的探针传的是 `self._support(attention_mask)`——**query 侧的支撑集**（并集）；
- `_check_exp_range` 拿到的是 `he_softmax` 里逐密文的 `attention_mask[i]`——
  **真正会把槽乘成 0 的那张**。

`softmax.py:151` 紧接着 `he_exp` 就是 `rescale(multiply(ct, mask))`。
所以 **38.1 落在这张 mask 清零的槽里，下一行就被乘成 0**。
guard 只看存活的槽，看到 10.203，窗口 `[-27.25, 21.73]`，宽松得很。

> **`he_exp` 窗口这条线到此为止。** 这正是 `bee4bcf` 把 guard 改成 mask-aware 的目的，
> 它第一次真正跑在设备上就把这条线关掉了。我的排除表里那个 ⚠️ 可以划掉。

---

## 2. 日志里只有**一次** `he_inv`，而且它被拒了

你的 §2 读成"第一次通过"、§3 读成"第二次被拒"。**是同一次。**

`_check_inversion_range` 的顺序是：

```python
if _debug():
    print(f"[range] he_inv observed [{low}, {high}] ratio ... against epsilon ...")
if low >= epsilon and high <= 1.0:
    return
raise ValueError(...)
```

**先打印，再判定。** 所以一次调用产生一行 `[range]` **加**一个 `ValueError`。
两处数字本来就是同一对：`[0.081449, 1.0897]` 和 `[0.08145, 1.09]`。

另外 §2 说"ratio=0.075 > epsilon=0.047，通过"——`ratio` 是 `low/high`，只是个诊断量，
不参与判定。判定是两条：`low >= epsilon`（0.0814 ≥ 0.0469 ✅）**和 `high <= 1.0`
（1.0897 ≤ 1.0 ❌）**。**它是栽在上界。**

于是 §3、§4、§6.2、§6.3 里建立在"有两次调用"之上的推理都不用再追了。
`07d` / `07e` / `inv_input_lift` 没出来，是因为**第一次** `he_inv` 就抛了。

---

## 3. `1.0897` 是精确的算术：`0.3632 × 3`

你的 §5 报了 `07c.denominator max 0.3632`。guard 检查的是 **lift 之后**的值
（`numeric.py` 里顺序是 `multiply(denominator, lift)` → `_check_inversion_range`，
docstring 写了"The range check runs *after* the lift"）。

```
0.3632 × 3 = 1.0896      guard 报的是 1.0897
```

**严丝合缝。不是 bootstrap，不是迭代，就是 lift 把它顶过了 1。**

### 3.1 我上一份的"别试 lift 2"是错的，撤回

上一份我扫了 lift × bootstrap 误差，得出"单调，lift 越大越好"，并写了
"**不要试 `--inverse-lift 1` 或 `2`**"。**那张表是在 `top = 0.3036`（ClearEngine 的
`07c`）下扫的，而在那个 top 下 lift 3 还在悬崖以内（0.911 < 1）。**

**天花板不是常数，是 `1 / max(denominator)`，而设备的 max 不等于本机的：**

| | `07c` max | 天花板 `1/max` | lift 3 落在 |
|---|---|---|---|
| ClearEngine | 0.3036 | **3.29** | 0.911 ✅ |
| 设备（`633ede0` 那次） | 0.3169 | 3.16 | 0.951 ✅ 勉强 |
| **设备（`20032f8` 这次）** | **0.3632** | **2.75** | **1.090 ❌ 越界** |

重扫，这次两个 top 都扫，并且把 guard 关掉好让越界的那格能跑出数来：

```
top=0.3036   (lift × top；* = 越过悬崖)
  lift 1  (0.3036)   err0 4.138e-07   err.017    32.27   err.05  2.256e+09
  lift 2  (0.6072)   err0 4.329e-05   err.017    1.306   err.05      299.5
  lift 3  (0.9108)   err0 3.073e-06   err.017   0.6103   err.05      20.77

top=0.3632
  lift 1  (0.3632)   err0 4.138e-07   err.017     19.4   err.05  8.536e+08
  lift 2  (0.7264)   err0 4.329e-05   err.017    1.106   err.05      170.8
  lift 3* (1.0896)   err0     190.3   err.017    973.1   err.05  1.043e+05
```

**看 `err0` 那一列**：越过悬崖之后，`--inverse-lift 3` 在**零 bootstrap 误差**下
就已经是 **190.3**。这和精度无关，提多少 bit 都救不回来——**变号就是变号。**

> **规则是"取放得下的最大值"，不是"取最大值"。**

### 3.2 所以：跑 `--inverse-lift 2`

设备天花板 2.75，lift 2 给 0.726，留 27% 余量。零误差下 4.3e-05，
标称 0.017 下 1.106。

一个提醒：**lift 2 在 `07c` max < 0.5 之前安全**。设备这个量在两次跑之间
从 0.3169 漂到 0.3632（每次跑密钥是新生成的，噪声不同），所以它是会动的。
`07c` 一旦过 0.5，lift 2 也会越界——那时 guard 会直接告诉你该用几（见 §4）。

### 3.3 顺带解释你的 §4

你问"1e+12 是迭代放大的还是输入带来的"。**两次跑不是同一个原因：**

- `633ede0` 那次：`07c` = 0.3169，×3 = 0.951，**没越界**。那次是精度（上一份的 §3）。
- `20032f8` 这次：`07c` = 0.3632，×3 = 1.090，**越界了**。这次是悬崖。

`07c` 在 0.30–0.37 之间漂，而 lift 3 把安全边际吃得只剩几个百分点，
**于是它有时候在界内、有时候在界外**。lift 2 把这个不确定性整个去掉。

---

## 4. 按你的 §6.4 改了两处

**（a）guard 现在直接告诉你该用几。** 它手里有 `high` 和 `lift`，
未 lift 的最大值就是 `high / lift`，天花板就是它的倒数取整：

```
... The maximum is over 1, where `2 - k*b` changes sign and the iteration inverts
instead of converging - at lift 3 that is 0.3632 x 3. Below the ceiling more lift
is strictly better, so the value to use here is the largest one that still fits:
`--inverse-lift 2`.
```

**（b）`--check-ranges warn`。** 你说得对：第一个 guard 一抛，后面全没了。

```
--check-ranges          # = raise，默认行为，拿答案的跑法
--check-ranges warn     # 打印同样的诊断然后继续，拿探针的跑法
--check-ranges off      # 不开（默认）
```

`warn` **不是默认**，因为没有 guard 的迭代返回的是"有限的、看着合理的、错的"数——
这套机制存在的全部意义就是拦这个。

---

## 5. 请跑这三个

**A（一条命令，最重要）：`--inverse-lift 2`。** 其他参数不变。
预期 guard 通过（0.726 < 1），跑完一层，`07d` 应该回到 0.05 量级。
`--inverse-lift` 不改 plaintext-cache tag，热 cache 直接跑。

**B：`--check-ranges warn --inverse-lift 3`**，把上次被 guard 挡住的探针补齐：
`07.inv_input_lift3` 的 carried max、`07d`、`07e`、`11a`/`11b`。
这一跑是为了**对照**：同一次跑里同时有"越界的 lift 3"和 A 的"界内的 lift 2"，
两边一减就知道悬崖占了多少、精度占了多少。

**C：上一份的那条 pytest，还没跑。** 它是唯一还没有数的量，而且不用跑 bench：

```bash
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q \
    tests/test_bootstrap_noise_level.py::test_how_wrong_the_bootstrap_is_on_a_full_vector
```

就算 A 成功了它也要跑：悬崖修掉之后，剩下的误差全是精度，而 §3 的表说
0.017 在 lift 2 下仍有 1.1 的相对误差。**这个数决定接下来要不要上 Meta-BTS**
（按 `--time-ops` 只要 +0.76 s/层，速度上完全不是问题）。

---

## 6. 我这边改了什么

| 改动 | |
|---|---|
| `numeric.py` | guard 越界时报出该用的 `--inverse-lift N` |
| `numeric.py` | `_range_violation()`：三个 guard 统一走它，支持 raise / warn |
| `bench.py` | `--check-ranges {off,raise,warn}`（`--check-ranges` 裸写 = raise） |
| `test_division_padding_floor.py` | 三个测试：guard 报出正确的 lift；天花板以下单调、以上是悬崖（零误差 190 vs 3e-06）；warn 模式继续执行 |

没有改任何算法。
