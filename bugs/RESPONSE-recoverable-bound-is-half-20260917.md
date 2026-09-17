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

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
