# 不是 bootstrap 放大 100x：×32 是构造；lift 已顶满，剩下的是 bootstrap 精度差一个数量级

日期：2026-09-22。回应 `633ede0`。参考数据来自本机 ClearEngine 同参数单层实跑
（`--depth 41 --layers 1 --limit 1 --per-stage --refresh-after-dense
--residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 --inverse-lift 3`），
**这是之前一直缺的那一列**。

---

## 0. 并排

| probe | ClearEngine（精确） | 设备 | 比值 |
|---|---|---|---|
| `07a0` carried max | **0.319** | **0.319** | **1.000** |
| `07a0` padding max | **0.3807** | **0.3807** | **1.000** |
| `07a0` \|x\| p50 | **0.02713** | **0.02713** | **1.000** |
| `07a` carried max | 10.21 | 38 | 3.7x |
| `07a` padding max | 12.43 | 16.04 | 1.3x |
| `07b.exp` max | 0.1056 | 0.2015 | 1.9x |
| `07c.denominator` max | 0.3036 | 0.3169 | **1.04x** |
| `07d.inverse_denominator` max | **0.04903** | **8.631e+05** | **1.8e+07** |
| `07e` carried max | **0.3058** | **1.089e+12** | 3.6e+12 |
| `11a.variance` max | 0.3924 | 8.841e+07 | 2.3e+08 |
| `11b.inverse_sqrt` max | 2.93 | 2.317e+154 | — |

---

## 1. bootstrap 没有放大 100x

`07a0` 与设备**逐位相同**——carried 0.319、padding 0.3807、p50 0.02713，三个数字全中。
bootstrap 的输入两边一样。

`07a0 → 07a` 的 ×32 是**构造出来的，不是误差**，`stage_07_softmax` 两行：

1. pack/unpack 把两个实密文塞进一个复密文的实部虚部，出来是 `x + conj(x)` = **2·Re**
   和 `i(conj(x) − x)` = **2·Im** —— **×2**，函数 docstring 第一段就写了这件事；
2. `restore = int(score_refresh_scale)` = **16**，整数乘，不耗 level。

**×2 × 16 = ×32。** 0.319 × 32 = 10.2，正是 ClearEngine 的 carried max 10.21。

119x 这个数还有一层：`magnitude_probe` 对密文取 `np.real(decrypt(ct))`，而 `07a0` 探的是
**打包后的复密文**，所以它只报了实部那一半；`07a` 探的 `refreshed` 是 8 个实密文，
实部虚部都在里面。两边测的不是同一批槽。

**但设备在 carried 槽上确实高 3.7 倍**（38 vs 10.21），这条是真的，见 §3。

---

## 2. `07d` 不是"基本正常"——它是整条链上唯一真正断掉的地方

报告把 `07d max 8.631e+05` 记成"基本正常"。参考值是 **0.04903**，差 **1.8e+07 倍**。

而它的输入 `07c` 两边只差 **4%**（0.3036 vs 0.3169）。

> **第一次 `he_inv` 在输入正确到 4% 的情况下发散。** 这是唯一需要解释的现象，
> 其余全部是它的下游。

`07e` 的 1e+12 是**继承**的，不是新产生的。算一下：`update_inv_D` 里
`squared = square(2 · exp_u · inv_D · k)`，把设备自己的数代进去，
`(2 × 0.2 × 8.631e5 × 384)² ≈ 1.7e+16`，求和后落在 1e12–1e16——`07e` 的 1.089e+12 正在这个带里。

所以 **k=384 不是问题**：ClearEngine 这一步的探针名同样是 `07e.halved_denominator_k384`，
k 由 `delta` 决定，两边完全一致，而 ClearEngine 的 `07e` 是 **0.3058**（≈ `07c` 的 0.3036，
本来就该守在 0.3 附近）。k 相同、输入差 1.8e7 倍，输出差 3.6e12 倍——是输入的错。

---

## 3. 断点定位：lift 已经顶到天花板，剩下的全是 bootstrap

Goldschmidt 对 `[epsilon, 1]` 之外**不是不准，是变号**：修正项 `2 − k·b` 在 `b > 1` 时翻符号，
然后每步平方。所以 `he_inv` 的输入哪怕只有一个 carried 槽越过 1.0，整行就完了。

之前没有任何探针测过这个输入：`07c` 是 **lift 和 bootstrap 之前**的分母，
`inv_iter01_b` 已经走完一步了。**这就是一直缺的那个数。** 已加探针 `inv_input_lift{N}`
（`THORFHE_DEBUG` 下开），ClearEngine 当场给出：

```
[range] he_inv observed [0.0792919, 0.910867]  ratio 0.0871  against epsilon 0.046875
  [probe] 07.inv_input_lift3   level 24  min +0.07929  max +0.9109  carried max 0.9109  padding max 0.5
[range] he_inv observed [0.069349, 0.917375]   ratio 0.0756  against epsilon 0.0116777
  [probe] 07.inv_input_lift3   level 24  min +0.06935  max +0.9174  carried max 0.9174  padding max 0.5
```

我的第一反应是"lift 3 把分母顶到 0.911，离 1.0 只剩 8.9%，太激进"。**这个想法是错的，我扫了一遍，
它被自己的数据否掉了。** 固定 `top = 0.3036`（= `07c` 的实测最大值），扫 lift × bootstrap 误差，
报 1/D 的最大相对误差：

| boot err | lift 1 | lift 2 | lift 3 |
|---|---|---|---|
| 0.000 | 4.1e-07 | 4.3e-05 | 3.1e-06 |
| 0.005 | 0.5024 | 0.2212 | **0.1592** |
| 0.010 | 2.003 | 0.5024 | **0.2869** |
| **0.017**（设备标称） | 32.27 | 1.306 | **0.6103** |
| 0.030 | 5.97e+04 | 8.465 | **1.988** |
| 0.050 | 2.26e+09 | 299.5 | **20.77** |
| 0.090 | 5.80e+15 | 5.92e+05 | **1.30e+04** |

**每一行 lift 越大误差越小，单调。** 降 lift 换余量只会更糟。而 lift 的上限由悬崖决定：
`1 / 0.3036 = 3.29`，所以 **`--inverse-lift 3` 已经基本顶到天花板**。

剩下的预算就摆在这儿了：

| | |
|---|---|
| 分母区间（`07c`，真 checkpoint） | [2⁻⁶, 0.3036]，**动态范围 19.4 : 1** |
| lift 上限（悬崖 `b ≤ 1`） | **3.29** |
| lift 3 下最小 carried 分母 | 0.0468 |
| 设备标称 bootstrap 绝对误差 | 0.017 |
| **最小分母的信噪比** | **2.8** |
| → 1/D 相对误差 | **61%** |

**在 0.017 的误差下，即使 lift 顶满，第一次 `he_inv` 也已经是 61% 的相对误差。**
1/D 的误差对 bootstrap 误差近似线性（约 36 倍），所以需要的精度是个可以写下来的数：

| boot err | 1/D 相对误差 |
|---|---|
| 0.017 | 0.610 |
| 0.005 | 0.159 |
| 0.001 | **0.0365** |
| 0.0005 | 0.0186 |

> **没有任何 level-free 整数的摆法能救一个"底部只有噪声三倍"的分母。这条是 bootstrap 精度问题。**

### 3.1 设备的实际 bootstrap 误差远不止 0.017

但 0.017 也**解释不了** 1.8e+07。反过来解：设备 `07d` 的相对误差是
`8.631e5 / 0.04903 = 1.76e+07`，在上表里插值——

| boot err | 1/D 相对误差 |
|---|---|
| 0.09 | 1.30e+04 |
| **0.12** | **1.38e+06** |
| **0.15** | **6.75e+07** |
| 0.20 | 1.47e+10 |

**→ 设备在这个站点的等效 bootstrap 绝对误差约 0.13，是标称 0.017 的 8 倍。**

独立的第二个估计：`07a` 的 carried max 38 vs 参考 10.21，
误差 `(38 − 10.21) / 32 = 0.87`。站点不同、量级不同，**但方向一致：
设备 bootstrap 的实际误差比文档大一个数量级。**

这才是要查的东西。

### 3.2 逐迭代对照（第一次 `he_inv`，设备数据取自你 §3）

| iter | Clear `b` min / max | 设备 `b` min / max |
|---|---|---|
| 01 | +0.0767 / **+0.274** | +0.0519 / **+1.608**  `>1 1/32768` |
| 02 | +0.0128 / +0.0257 | −1.213 / +0.0476 |
| 03 | +0.00331 / +0.00372 | −0.032 / +56.67 |
| 04 | +0.00389 / +0.00390 | −1.015e+06 / +0.0039 |

ClearEngine 的 `b` **从不越过 0.274，从不为负**。设备在 **iter01 就已经有一个槽在 1.608**。
第一个坏步是 **iter01，不是 iter03**；报告说"iter03_b 的 max=56.67 是个别槽，不影响整体"——
在这个迭代里一个越界的槽就是会整行发散，56.67 是 1.608 传下来的第三代。

---

## 4. `%0` 的位置信息

`worst at` 的索引解出来（`n_slot=16`，`group_size=2048`，探针把 8 个密文首尾相接）：

- `07a`：196608 / 163840 / 229376 = **6·32768 / 5·32768 / 7·32768**
  → **第 5、6、7 号密文的第 0 槽**。那三个正是 `refreshed[index + half]`，即**虚部那一半**。
- `07e`：6144 / 28672 / 30720 = **3·2048 / 14·2048 / 15·2048** → **每个 group 的第 0 槽**。
- `07d`：slot 0。

误差全部落在"某个密文/某个 group 的第 0 槽"，不是随机分布。这不像噪声，像
conjugate / rotate-and-add 归约在 slot 0 上的结构性问题。记在这里备查，但**先验证 §3**，
因为 §3 的代价是零。

---

## 5. `_check_exp_range` 为什么没触发——是我的 bug，已修

三个 guard（`_check_exp_range`、`_check_inversion_range`、`_check_invsqrt_range`）都写成

```python
if not (self.check_ranges and getattr(self.engine, "inspectable", False)):
    return
```

而 `inspectable = True` **只在 `ClearEngine` 上有**（`grep -rn inspectable` 全仓只有这一处赋值）。
设备引擎没有这个属性 → `getattr(..., False)` → **三个 guard 在硬件上从来没有执行过一行**。

写的时候的想法是"设备读不回密文"。这是错的：`magnitude_probe` 一直在设备上做
`engine.decrypt(ct)`。它们读得回，只是要花一次解密。而**硬件恰恰是值会越界的地方**。

已加 `--check-ranges`：把 `engine.inspectable = True` 设到引擎对象上
（`instrument()` 早就在用同样的方式给 pybind11 对象绑属性，所以这条路是通的）。
代价：每次 guarded 调用一次解密，一层约十来次，GV100 上 76 ms/次 ≈ **1 秒/层**。
所以是 opt-in，不默认开。

---

## 6. 请跑这两个（A 优先，一条 pytest）

**A. 直接测 bootstrap 的绝对误差**——这是现在唯一重要的数，而且**不用跑 bench**。
测试已经写好了：

```bash
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q \
    tests/test_bootstrap_noise_level.py::test_how_wrong_the_bootstrap_is_on_a_full_vector
```

它在 bench 的真实参数下加密一个**稠密**向量（量级就是 `he_inv` 拿到的
`[2⁻⁶, 0.3036] × 3`），自举一次，打印 max / p99 / p50 和**误差最大的 5 个槽的 index**。
`-s` 不能省，它只打印不断言。

之所以要重测：这个文件里其他探针用的 `_values` 只有 **3 个非零槽**（65536 分之 3），
标称的 0.017 就是那么测出来的。**3 个槽看不出"误差只落在特定位置"的模式**，
而 §4 说的正是这种模式；也看不出稠密向量和稀疏向量的差别，而 `he_inv` 自举的是稠密分母。

- max ≈ 0.017 → §3.1 被否，问题不在 bootstrap 精度，转 §4 的 slot-0 结构线；
- max ≈ 0.1 或更大 → **§3.1 成立，根因定案**；
- 不管哪种，`worst` 那一行的 `%16` / `%2048` 如果也全是 0，§3 和 §4 就并成一条线了。

**B. 带 `THORFHE_DEBUG=1 --check-ranges` 跑一层**，报两个数：
`07.inv_input_lift3` 的 **carried max**，和 `07.inv_iter01_b` 的 **`worst at`**
（上一份报告里这行被截掉了，而它直接说出是哪个槽先越界）。
carried max > 1.0 的话 `ValueError` 会当场抛在 `he_inv` 那一行，不用再往下找。

**不要试 `--inverse-lift 1` 或 `2`。** §3 的表已经扫过了：单调，更小的 lift 严格更差。

---

## 7. 我这边改了什么

| 提交 | 改动 |
|---|---|
| 本次 | `numeric.py`：`he_inv` 在迭代开始前探 `inv_input_lift{N}`（`THORFHE_DEBUG` 下），带 support 分列 carried/padding |
| 本次 | `bench.py`：`--check-ranges`，把 `inspectable` 设到设备引擎上，让三个 guard 在硬件上真正生效 |
| 本次 | `test_division_padding_floor.py`：两个测试锁住 §3——lift 单调、顶 3.29、0.017 下仍 61%；以及 range check 在 bootstrap 之前所以看不见悬崖 |
| 本次 | `test_bootstrap_noise_level.py`：`test_how_wrong_the_bootstrap_is_on_a_full_vector`——稠密向量上的自举绝对误差 + argmax 位置 |
| 本次 | 本报告：ClearEngine 同参数参考列 |

没有改任何算法。

---

## 8. 我在这份里改过口的地方

- "lift 3 太激进，离悬崖只剩 8.9%，降到 1 或 2 试试"——**扫表否掉了**，单调，lift 越大越好，
  3.29 是天花板。原来的报告名和 §6 的 B 项都是按这个写的，已经改掉。
- 顺带否掉的还有一条：0.09 的 bootstrap 误差在 lift 1 下也一样炸（5.8e+15），
  因为分母底部 2⁻⁶ 会被推成负数。**两头都是悬崖，中间没有安全的 lift。**
