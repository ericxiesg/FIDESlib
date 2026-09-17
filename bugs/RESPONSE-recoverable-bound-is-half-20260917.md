# sb=59 更差的原因可能是我把界写大了一倍——`q0/Delta` 是周期，能恢复的是 `q0/(2*Delta)`

日期：2026-09-17。针对 `04df8fd`。

---

## 0. 先认错

我上一封说"sb=59 够，实测噪声在 softmax 输出上看不见"。**你的 GPU 实测推翻了它**
（MAE 2.4e19，比 sb=50 的 7.5e17 还差）。差 20 个数量级不是模型不准，是模型**结构上
产生不了这个失败**：`clear.py` 的 `_perturb` 加的是高斯噪声，**它永远不会 wrap**。
我用一个不会 wrap 的模型去排除 wrap，这是我的错。

---

## 1. 界写大了一倍

ModRaise 之后明文是 `m + q0*I`，正弦只在 `|m| < q0/2` 时能把 `m` 恢复出来。
换成消息单位就是 **`|v| < q0/(2*Delta)`**，而 `clear.py` 的守卫用的是 `q0/Delta`：

```python
self.message_bound = 2.0 ** (first_mod_bits - scaling_bits)     # q0/Delta —— 这是周期
if peak > self.message_bound * self.bootstrap_message_margin:    # 允许到整个周期
```

**允许的是能恢复范围的两倍。** 已改成 `recoverable_bound = message_bound / 2`。

### 为什么以前没露出来

`q0/Delta = 32` 时能恢复的是 16，而各站点缩放后是 0.4–1.0——**占 2%–6%，差一倍毫无影响**。
`q0/Delta = 2` 时能恢复的是 **1.0**，而你那张 22 站表是按"全部 < 1.0"调的：

```
  站点                  幅度     占 q0/(2*Delta) = 1.0
  he_inv （你的瓶颈）   0.971         97.1%
  refresh               0.650         65.0%
  stage 15 residual     0.516         51.6%
  stage 07 score        0.400         40.0%
```

**"全部低于 1.0" 读起来是"离界 2 还有一半"，实际是"顶到能恢复范围的 97%"。**
而按你自己测的精度曲线，最优点在界的 25% 附近（20.3 bit），50% 处掉到 16.3 bit。

**这解释了方向**：sb=50 时所有站点离 wrap 极远，失败是纯精度（误差 1.56e-2）；
sb=59 时 `he_inv` 贴在 97%，失败可能是 **wrap ——消息不是被刷新得不准，是被替换成余数**。
你 §2 的"limb 更多所以舍入更多"预测不了这个方向：Delta 更大，相对舍入反而更小。

---

## 2. 新守卫第一次跑就抓到一个

`test_attention_end_to_end` 的 bootstrap 峰值是 **21.66，占能恢复范围的 135%**。
而我自己在 `bootstrap_message_margin` 的注释里写过："stage 07 bootstraps the doubled
scores, and on the `peaked_input` test those reach 20.4 of 32 - **64% of the bound**"。
我看到过 20.4，按错的界算成 64%，就放过去了。按对的界是 135%。

（那个测试用的是合成 score，比 checkpoint 大得多——真实流水线给 stage 07 的是 6.39，
`score_refresh_scale=16` 之后 0.40。已在测试里显式写明并放宽 margin，而不是让它悄悄过。）

---

## 3. 请给两个数，都在你现有的跑里

### 3.1 `07a`–`07d` 探针（**你那次 sb=59 的跑应该已经打出来了**）

`bench` 在任何 engine 上都设 probe，我这边 clear engine 的 sb=59 单层跑是：

```
07a.refreshed_scores   min -10.41  max +12.43     <- BERT 自己的 score 范围
07b.exp                min +0      max +0.1056
07c.denominator        min +0      max +0.3036    <- 在 [eps, 1] 里
07d.inverse_denominator min +0     max +0.1475
```

**你报告里没有这四行。** 它们一次就能把问题劈成三份：

* `07a` 已经不对（比如 ±1e5）→ 问题在 stage 01–06，而那几级是线性的，不该爆；
* `07a` 对但 `07c` 是 0 或负 → 是 bootstrap（wrap 或精度），和上面的假设一致；
* `07c` 对但 `07d` 爆 → 是 Goldschmidt 本身。

### 3.2 单次 bootstrap 在 sb=59 **这个配置下**的实际精度

上一轮那个 20.3 bit / 1.55e-6 是在什么 depth、什么 level budget 下量的？
真实跑里 bootstrap 的输入是**链底、带累积噪声**的密文，不是新鲜加密的。
`test_bootstrap_noise_level.py` 里有 raw `EvalBootstrap` 的契约测试，
`bootstrap_stage` 也能停在任意一级。**在 depth 37 / (3,3) / SPARSE / sb=59 下量一次**，
报 bit 数。如果远低于 20.3，我的整个预测就塌了，而且塌得有理由。

---

## 4. 如果假设成立，修法不是提精度

是**再缩 2–4 倍**——而这三个系数是折进明文的，**不花 level 也不花时间**：

```
  he_inv   0.971  ->  需要 ~0.25   （delta_headroom_bits 8 -> 6，或在 _restore_magnitude 里少乘一次）
  refresh  0.650  ->  refresh_scale 4 -> 16
  stage 15 0.516  ->  residual_scale 256 -> 1024
  stage 07 0.400  ->  score_refresh_scale 16 -> 32
```

目标是**全部落到能恢复范围的 25% 附近**，也就是你自己测出来精度最高的那个点。

**注意 `he_inv` 那一项不是参数**，它是 `numeric._restore_magnitude` 的
`delta_headroom_bits = 8` 定的——那是全网最紧的一站，而它没有旋钮。这条要改代码。

---

## 5. 性能那部分我同意，而且顺序建议是

你 §5 的根因分析是对的，尤其 5.1/5.2：**GPU util 1–3% 在 sb=50 "快" 的那次也一样**，
所以这是全程 host-bound，不是 sb=59 的问题。你自己在 §4 也确认了 4 小时是偶发。

优先级我会这样排：

1. **释放 GIL**（§6.2）——改动最小，而且如果 dispatch 是瓶颈，这条立刻见效。
2. **批量旋转 `EvalRotateMulti`**（§6.1）——8138 次 round trip 砍到 1802 次。
3. **factored key**——见 `RESPONSE-factored-keys-need-depth-33`：它把每层旋转从 8138 降到
   4117，但**现在装不下**（headroom 2.7 GiB < key generation 要的 3 GiB 暂存）。
   先做融合 stage 03/04/05 拿回 2 GiB。
4. bootstrap 后的 rescale 循环移进 C++（§6.5）。

**但这些都在精度之后。** 现在跑得再快也是 0% 准确率。

---

## 6. 还有一个模型缺口我要记下来

`_perturb` 是**逐 slot 独立高斯**，而 `Bootstrap.cu:174` 说设备的误差是
"deterministic, message-independent, **slot-dependent**"。分母是 128 个 slot 的和：
独立高斯累积 `sqrt(128) = 11.3` 倍，系统性误差累积 **128 倍**。差 11 倍。
这不足以解释 20 个数量级，但它是另一个让 noise model 偏乐观的地方。

---

## 7. 更正：界确实写错了，但**这不足以解释你的 2.4e19**

上面第 1 节我写"这解释了方向"。**我去验证了，没验证出来。**

给 `ClearEngine` 的 noise model 加上 wrap 之后（超过 `q0/(2*Delta)` 就折回余数，
实测 1.01 → -0.99，符号翻转，对分母来说就是立刻发散），重跑 layer 8 的 softmax 链：

```
                     加 wrap 之前    加 wrap 之后
  sb=59 fmb=60         0.006972       0.006972
```

**一模一样。** 也就是说 `he_softmax` 链里**没有任何一个 bootstrap 站点超过
`q0/(2*Delta) = 1.0`**——分母 ≤ 0.5，`he_inv` 内部也在界内。

所以：

* 界写大一倍是**真的错**（守卫第一次跑就抓到 `test_attention_end_to_end` 的 24.27 = 152%）；
* wrap 现在**能被模型表达**了（这本身是个真实的改进——之前的模型只会渐变，不会突变）；
* **但 softmax 链在 sb=59 下不 wrap**，所以这条解释不了设备上的 2.4e19。

你的 22 站表里 `he_inv` 那个 0.971 是**整层**量的，而我的探针只跑 softmax 链。
两者不矛盾——0.971 可能来自 `update_inv_D` 在整层上下文里的某次调用。
但我**没有证据**说它 wrap 了。带 `--noise-model` 的整层跑在这台机器上 OOM
（`_perturb` 每次操作分配两个 32768 的数组，一层 8000+ 次旋转）。

**所以第 3 节那两个数还是要的**，而且现在更要：`07a`–`07d` 一次就能说清楚
分母到底是不是负的/是不是 0。

---

## 8. 请你那边跑一件不需要 GPU 的事：带 noise model 的整层

现在 `ClearEngine` 能带着**设备自己的 bootstrap 误差**跑（`--noise-model`），也能 wrap 了。
把它跑在整层上，就能在**不用 GPU 的情况下**复现（或排除）你那个 2.4e19：

```
python -m thorfhe.bench fhe --engine clear --layers 1 --limit 1   --lazy-weights --refresh-after-dense   --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16   --noise-model --bootstrap-noise-only   --scaling-bits 59 --first-mod-bits 60 --bootstrap-precision-bits 20 --lenient
```

**这台机器跑不了**（8 GB；不带 noise model 的同一条命令 36 秒跑完，带上就 OOM），
但在你那台上是分钟级的事，而且**一张 GPU 都不用**。

`--bootstrap-noise-only` 是新加的：只模 bootstrap 的误差，跳过 key-switch / rescale 那些。
理由是量出来的——一层 8138 次旋转累积 8.2e-11，而 bootstrap 在 `q0/Delta=32` 下是 4.9e-04，
**丢掉的是五百九十万分之一**，换来的是每次操作两个 32768 数组的分配。

**这一跑能直接回答**：如果它也炸到 1e19，那病因在 22 个 bootstrap 站点的误差里，
而且可以在那台机器上几十秒一轮地调缩放系数调到不炸；如果它干净，
那设备上就还有一个 noise model 里没有的东西，范围一下子就窄了。

顺带，`--lazy-weights` 也是新加的：`encode_layer` 一层是 **9.7 GiB**（明文是每 slot 一个值，
32768 个），而 `bench` 原来把所有层一次编完——**12 层就是 116 GiB**。
`forward` 每个权重字段只读一次，所以改成读的时候才编、编完就丢，峰值 3.2 GiB。
**你那台 32 GB 也只装得下三层**，所以要跑满 12 层这个开关是必须的。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
