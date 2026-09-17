# 修正 softmax scale 后 22 站重测：三参数验证通过

日期：2026-09-17。针对 `1f0053d`。

---

## 1. 基线（三参数全 = 1，修正后 softmax scale = 1/16）

```
site                                          n      min       max    % of bound 2
layernorm.py:211 (stage 15)                   4    100.2     129.7     6484%   ❌
softmax.py:192 (stage 07)                     4    6.648     7.967      398%   ❌
layernorm.py:191 (refresh)                    4    0.862     1.396       70%   ⚠️
softmax.py:78 (he_inv update_inv_D)           1    0.971     0.971       49%   ⚠️
feedforward.py:86 (stage 13 gelu)             8    0.218     0.286       14%   ✅
softmax.py:106 (he_inv he_softmax)            1    0.001     0.001        0%   ✅
```

**修正 scale 后 refresh 从 1.91-2.60 降到 0.86-1.40**——温度正确后注意力权重变小了。
stage 15 和 stage 07 不变（不受 softmax 温度影响）。

---

## 2. 三参数 = (256, 4, 16) 验证

```
residual_scale=256  refresh_scale=4  score_refresh_scale=16
```

```
site                                          n      min       max    % of bound 2
softmax.py:78 (he_inv update_inv_D)           1    0.971     0.971       49%
layernorm.py:211 (stage 15)                   4    0.392     0.507       25%   ✅
softmax.py:192 (stage 07)                     4    0.416     0.498       25%   ✅
layernorm.py:191 (refresh)                    4    0.215     0.349       17%   ✅
feedforward.py:86 (stage 13 gelu)             8    0.218     0.286       14%   ✅
softmax.py:106 (he_inv he_softmax)            1    0.001     0.001        0%   ✅

largest 0.9712; 22/22 < 1.0, 20/22 < 0.5
```

---

## 3. 三参数 = (256, 2, 16) 对比

```
residual_scale=256  refresh_scale=2  score_refresh_scale=16
```

```
site                                          n      min       max    % of bound 2
softmax.py:78 (he_inv update_inv_D)           1    0.971     0.971       49%
layernorm.py:191 (refresh)                    4    0.431     0.698       35%
layernorm.py:211 (stage 15)                   4    0.392     0.507       25%
softmax.py:192 (stage 07)                     4    0.416     0.498       25%
feedforward.py:86 (stage 13 gelu)             8    0.218     0.286       14%

largest 0.9712; 22/22 < 1.0, 17/22 < 0.5
```

---

## 4. 两种 refresh_scale 对比

| refresh_scale | refresh max | 占界 | 精度（插值） |
|---:|---:|---:|---|
| 2 | 0.698 | 35% | ~18-19 bit |
| 4 | 0.349 | 17% | ~19-20 bit |

精度曲线：5% = 17.3 bit，25% = 20.3 bit，50% = 16.3 bit。
17% 和 35% 在最优点两侧，**4 略优**（17% 更接近 25% 最优点）。

---

## 5. 瓶颈是 he_inv 的 0.97（49%）

`softmax.py:78` 的 `update_inv_D` 里 `he_inv` 的 bootstrap 输入 = 0.9712，
**不受三个参数中任何一个影响**，是所有 22 站里最大的。

49% 按精度曲线大约 16-17 bit。这是当前架构的精度上限——
除非改 `he_inv` 的实现（比如把分母的 bootstrap 也缩放），否则 22 站的精度
由这个站点决定。

---

## 6. 建议

```
residual_scale       = 256    0.507  (25%)   ✅ 最优
score_refresh_scale  = 16     0.498  (25%)   ✅ 最优
refresh_scale        = 4      0.349  (17%)   ✅ 接近最优
```

三个系数都定了，22/22 站 < 1.0，20/22 站 < 0.5。
**可以迁移到 sb=59/fmb=60 跑 GPU 端到端了。**
