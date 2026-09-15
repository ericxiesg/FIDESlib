# softmax 之后全错：分母很可能掉出 `he_inv` 的定义域

日期：2026-09-15。起因：远端报告 q/k/v/scores 的 MAE 都是 1e-7，**softmax 之后全错，
准确率 50%（随机）**。本文给出一条可以当场证伪的具体假设，以及为验证它补的检查和测试。

---

## 1. 结论先行

`he_softmax` 里第一次 `he_inv` 要求分母落在 `[inv_epsilon, 1]`，layer 0 的
`inv_epsilon = 2^-11 = 4.883e-4`。而你的探针报的是

```
[probe] 07c.denominator          max +0.0005854
```

**那是全槽最大值**，只有 `inv_epsilon` 的 1.2 倍——所以**大多数槽必然在下界以下**。

低于下界时，Goldschmidt **不发散，而是饱和**：迭代次数和每步的 `k` 都只由 `epsilon` 推出，
分母更小就等于迭代表不够用，答案停止增长。在 `ClearEngine` 上实测（`epsilon = 2^-11`）：

| D / epsilon | 1/D 真值 | he_inv 返回 | 相对误差 |
|---:|---:|---:|---:|
| 4 | 512 | 511.98 | 4e-5 |
| 1 | 2048 | 2047.9 | 5e-5 |
| **0.5** | 4096 | 3946.9 | **3.6%** |
| **0.25** | 8192 | 6392.6 | **22%** |
| 0.1 | 2.05e4 | 9019.8 | 56% |
| 0.001 | 2.05e6 | 11555 | 99.4% |

**它返回一个有限的、看起来合理的错值**，不报错、不溢出，上限约 11500。
一个建立在这上面的 softmax 会给出"看起来像权重的权重"，然后按随机水平分类——
**这正是准确率 50% 的形状**。

---

## 2. 一条能当场证伪的检查

拉最新的 bootstrap-dev 重跑一次带探针的，看 `07c` 这一行。我把探针改成报分位数了
（`bench.py`，先前那条 commit），所以现在会直接打出 `p50`：

```
[probe] 07c.denominator   ...  |x| p50 ...  p99 ...  max ...
```

**判据很简单：`p50` 如果低于 `4.883e-4`，就是这个问题。**

* `p50 ≥ 4.883e-4` → 假设作废，我另找。
* `p50 < 4.883e-4` → 上表给出误差量级，且越小越糟。

顺带也请看 `07b.exp` 的 `p50`：分母是 exp 在组内的和，分母偏小的直接原因就是 exp 偏小。

---

## 3. 如果成立，问题出在哪

**不在 `he_inv`，在标定。** THOR 的 `attention_key_scale`（layer 0 是 `1/512`）
存在的唯一目的就是把分母放进这个窗口。`thorfhe/softmax.py` 的 `calibrate()` 文档里写着：

> 上界之上，15 次多项式不再是指数、很快发散；
> **下界之下，Goldschmidt 被要求求一个它没有设定去求的范围，返回的是无意义的值。**
> 有用的窗口大约是分母的三个数量级。

所以如果 `p50` 掉在窗口外，要改的是喂给 softmax 的 score 幅度，不是迭代本身。
两条路：

1. **用真实激活重新标定**，像 `calibrate()` 做的那样——THOR 的常数是在**它自己的**
   模型和数据上标的，我们的 q/k 路径不一定落在同一个窗口；
2. **先确认我们的 score 幅度和 THOR 的一致**。你的探针有
   `07a.refreshed_scores min -10.45 max +12.45`，而 NARROW 的窗口是
   `min_x=-27.25, max_x=21.73`。**我们的 score 只占了那个窗口的中间一小段**——
   这和"exp 偏小、分母偏小"是一致的。这一条可能就是根因：不是 softmax 算错了，
   是**送进 softmax 的东西比标定时小**。

第 2 条特别值得先查，因为它便宜：把 `07a` 的 min/max 和 `min_x/max_x` 对一下即可。

---

## 4. 这次补了什么（`ClearEngine` 侧）

起因是 Part19 的评估，指出 `ClearEngine` 以为自己在检查、实际漏掉了几处。逐条核过
（其中一条它说错了，见 §4.5），补的都带测试。

### 4.1 `he_inv` 现在拒绝越界的分母

`numeric.py` 的 `_check_inversion_range`。只在能明文读值的引擎上生效
（`ClearEngine.inspectable`），设备上没有这个能力——设备侧对应的信号就是 §2 那个 `p50`。

只检查 `ones` 标出的槽，因为其余槽按构造是空的。

**它立刻在一个现有的、当时通过的测试上触发了**：
`test_softmax_is_a_distribution_on_a_peaked_input` 用的是生产参数 `Softmax.NARROW`，
背景 score 取 `[-2, 2]`，分母中位数 2.5e-4——**只有下界的 51%**。
它以前能通过，是因为精确算术加上后续 `update_inv_D` 的精化把一个饱和的初值救了回来。
**设备上这两个条件都不成立。** 已把背景 score 改成和另一个测试相同的 `[-8, 8]`，
测试通过，并在 docstring 里记了原委。

> 这件事本身就说明了问题：**生产参数配偏小的输入，会悄悄越界而所有断言照样绿。**

### 4.2 `multiply` 路径补上 scale 检查

`add`/`subtract` 一直有，`multiply` 一条都没有——实测能一路做到 `D^4`。
设备侧的对应断言是 `Ciphertext::mult` 的 `assert(NoiseLevel == 1)`（`Ciphertext.cpp:551,563`）、
`multPt` 的 `assert(NoiseLevel < 2)`（`:478`）、`multScalar` 的
`assert(this->NoiseLevel == 1)`（`:913`）——**都在 FIXEDAUTO/FLEXIBLE 分支之外，
FIXEDMANUAL 也会走到**，但它们是 `assert`，Release 编译里是空的。
而且 metadata 本来也表示不了：`ScalingFactorReal` / `ScalingFactorRealBig` 是
NoiseLevel 1 和 2 的缩放因子，**没有 `D^3` 的表项**。

整数乘**故意不查**：`multIntScalar` 不碰 metadata（`ApproxModEval.cu:131-136`），
`_restore_magnitude` 正是靠这一点。

### 4.3 `add_inplace` 丢 degree

`_binary` 算出了 `max(x.degree, y.degree)`，写回时漏了 `degree` 这一项。
于是 `x` 声称自己是 degree-1 却带着 c2，而 `rotate` / `multiply_1j` / `bootstrap`
——**恰好是设备上会静默丢掉 c2 的那三个**——全部放行。
这就是当初 1e124 那个失败，守卫被摘掉的版本。9 个调用点目前没踩到（两边 degree 恰好一致），
但那是巧合。已补，并加了测试。

### 4.4 bootstrap 的两个界

* **输入幅度**：`q0/Delta = 32`。超过就拒绝——ModRaise 留下 `m + q0*I`，
  正弦只在零附近近似模约简，贴着界的消息不是被刷新，是被替换掉。
* **`keep_levels` 欠额**：`Engine.bootstrap` 会抛，`ClearEngine` 以前无条件返回 `keep_levels`，
  所以"bootstrap 之后要 20 级但预算只给 18 级"的调度表能顺利通过、到设备上才炸。

### 4.5 一处 Part19 说得过头，我没照做

它建议阈值用 `message_bound / 4`。**那个 4 没有依据**，而且用它会把两个本来通过的测试判红。
改成硬界 `message_bound`，余量做成可调参数 `bootstrap_message_margin`（默认 1.0）。

不过那次误判本身量出了一个值得记的数：`test_softmax_is_a_distribution_on_a_peaked_input`
里送进 bootstrap 的峰值是 **20.4，占 32 的 64%**。`stage_07_softmax` 会把 score 加倍，
所以这个余量比看上去薄。**如果 §2 的判据不成立，这是我要查的下一个地方。**

### 4.6 噪声模型（可选，默认关）

`ClearEngine(noise_model=True)`。**不是** CKKS 噪声分析，只保证量级对：
新鲜加密 / rescale / key switch 各注入 `2^-42 ~ 2^-40` 的绝对扰动，
bootstrap 注入 `message_bound * 2^-22 = 7.6e-6`——**比其余的大七个数量级，
而且与被刷新的值的大小无关**。

这条正是为 `he_inv` 建的：它开头就 bootstrap 分母（`numeric.py:184`），
而分母只有 2e-4，绝对误差 7.6e-6 就是 **4% 的相对误差**。

**实测结果是这条假设不成立**：开噪声跑 `he_inv`，Goldschmidt 不变量
`b / b.delta` 在第 8 次迭代是 0.941，无噪声是 0.943——**没有发散**。
所以 bootstrap 精度不是原因，这条排除掉了。噪声模型留着，因为它是唯一能看见这类问题的工具。

### 4.7 其他

* `relinearize` 以前共享 numpy 数组，现在和别处一致地复制；
* `plan_rotations` 的 dry run 用零权重，值全是人造的，所以它关掉 §4.1 的范围检查——
  那里量的是**调度**，不是值。

---

## 5. 测试

新增 20 条，全绿（全量 108 passed / 46 skipped）：

* `tests/test_clear_engine_contract.py`（17 条）——上面每个洞一条，加噪声模型的量级排序；
* `tests/test_stage8_thor_numeric.py` 新增 3 条——`he_inv` 拒绝越界、
  **把饱和行为的实测数字钉住**（禁用检查后测）、以及 `[epsilon, 1]` 两端都要通过。

---

## 6. 要你做的

1. **`07c.denominator` 的 `p50`**，和 `4.883e-4` 比 —— 这是 §2 的判据，一行就能定。
2. **`07a.refreshed_scores` 的 min/max**，和 NARROW 的 `min_x=-27.25, max_x=21.73` 比 ——
   §3 第 2 条，同样便宜。
3. 顺带：`07b.exp` 的 `p50`。

这三个数出来，softmax 的事基本就定了。
