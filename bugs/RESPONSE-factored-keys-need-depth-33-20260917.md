# factored key 用不了的原因不是噪声，是 key generation 的 3 GiB 暂存；而解法正好是对齐 depth

日期：2026-09-17。针对 Part22 §3 和 §2.6。

---

## 0. 结论

| 问题 | 答案 |
|---|---|
| 为什么不能用 factored key | headroom 2.7 GiB < key generation 需要的 3 GiB 暂存，**在 AddRotationKeys 就 OOM，一个 stage 都跑不到** |
| §3.3 "精度没解决前先关掉 binary/factored" | **错了 6 个数量级**，而且关掉两个 = 210 把键 = 28 GiB，根本装不下 |
| factored 的 ~900 次旋转 | **实测 4117**。~900 在物理下界 1802 之下，不可能 |
| 改成 factored 后能对齐参数吗 | **反过来：对齐 depth 到 33 才能用 factored** |

---

## 1. key-switch 噪声这条理由，差 6 个数量级

`clear.py:111` 的模型：一次 key-switch 的噪声是 scale 以下 40 bit。N 次旋转按随机游走累积 `sqrt(N)`：

```
  一索引一键      1802 次  ->  3.86e-11
  factored        4117 次  ->  5.83e-11
  binary          8138 次  ->  8.20e-11
```

而 §3.3 担心的那个 11 bit bootstrap，误差是 **4.87e-04**（相对它自己的界）：

```
  8.2e-11 / 4.9e-04  =  1/5941767
```

**binary 和 factored 的差（2.4e-11）比 bootstrap 误差小六百万倍。**
这条不该进入决策。而且"两个都关掉"意味着 210 把键 28 GiB，32 GB 卡装不下——
这个建议本身不可执行。

---

## 2. ~900 次旋转是不可能的

一索引一键时每次旋转就是一步，所以 **1802 是物理下界**。
`RotationBasis.steps`：索引本身是键 → 1 步；能拆成基里两个键的和 → 2 步；否则退回二进制。
所以 factored 必然在 1802 和 3604 之间偏上。

`bench budget --extra-rotation-keys 6` 实测：**4117/layer**。
（我上一轮用 +4 个键测的是 4413，一致。）

---

## 3. 真正的原因：3 GiB 的 key generation 暂存

```
=== --binary-rotations ===
  rotation keys        2.5 GiB   15 keys
  bootstrap keys       9.2 GiB
  bootstrap plaintexts 7.6 GiB
  everything else      9.0 GiB
  total               28.3 GiB      headroom 3.7 GiB

=== --extra-rotation-keys 6 ===
  rotation keys        3.5 GiB   21 keys      <- +1.0 GiB
  total               29.3 GiB      headroom 2.7 GiB
  -- headroom is under the 3 GiB key generation needs for its decomposition scratch:
     expect the OOM in AddRotationKeys, before anything runs.
```

**注意 rotation key 只占 28.3 GiB 里的 2.5 GiB——9%。** 大头是 bootstrap 的
keys + plaintexts = 16.8 GiB（59%）。所以旋转策略从来不是显存的主要杠杆，
它只是**刚好卡在 key generation 暂存的门槛上**。

---

## 4. 对齐 depth 到 33，factored 就装得下

21 把键（factored）下的四种参数：

```
  depth 37  dnum  4    total 29.8 GiB   headroom 2.2 GiB   OOM in key gen
  depth 33  dnum  4    total 28.0 GiB   headroom 4.0 GiB   装得下  <--
  depth 37  dnum 17    total 72.8 GiB   headroom -40.8 GiB  装不下
  depth 40  dnum  4    total 31.2 GiB   headroom 0.8 GiB   OOM in key gen
```

**depth 33 省下 1.8 GiB，正好把 headroom 抬过 key generation 的 3 GiB 门槛。**

而 depth 37 vs 33 和"我们的 LN 内部不自举"是**同一个决定的两面**
（Part22 §2.6 自己写的）：CPU 在 LayerNorm 内部自举，每层 35 次而不是 22 次，
换来 post-bootstrap 12 而不是 20，于是 depth 33。

所以链条是：

```
LN1/LN2 内部加自举（+13 次/层）
  -> post-bootstrap 20 -> 12
  -> depth 37 -> 33
  -> headroom 2.2 -> 4.0 GiB
  -> factored key 装得下
  -> 每层旋转 8138 -> 4117（减半）
```

**一句话：不是"改成 factored 之后对齐参数"，是"对齐 depth 才能改成 factored"。**

### 但这是代码改动不是开关

我们的 `he_layernorm` 不自举（`layernorm.py:160` 的注释写着"No bootstrap: stage 10 leaves
enough levels"）。要到 depth 33 得在 LN 里加自举点，然后**重新推一遍 level 调度**——
33 是 CPU 自己的排布下的数，我们的排布不一样（我们在别处省了 level：
LN 内不刷新 −2、softmax 少一次 −1、context 后不刷新 −2、GELU 只刷一次 −8）。
所以 33 是个待验证的目标值，不是给定值。要先量深度最深的那条链。

---

## 5. 另外三个参数：对不齐，也不该对

| 差异 | 能不能对齐 | 为什么 |
|---|---|---|
| dnum 4 vs 17 | **不能** | 一把键是 `2*dnum` 个多项式；dnum 升高 K 会降，但 `2*dnum` 涨得更快：`(34*41)/(8*49) = 3.56x`。21 把键 + bootstrap 键从 11.7 变 56 GiB，总计 72.8 GiB。CPU 能用 17 是因为它**从磁盘流式**（3.26 TB 逻辑读），那是它换掉显存的方式 |
| SPARSE vs UNIFORM | **不能（配 factored 时）** | +3 层自举深度 → depth 40 → headroom 0.8 GiB，同样在 key gen OOM。而且这换的是安全裕度不是正确性 |
| FIXEDMANUAL vs FLEXIBLEAUTO | **不该** | 省掉每层 ~300 次明文乘（是时间不是显存），但会废掉整个显式 level 契约——`ClearEngine` 的 `_require_canonical_scale`、`ScaleMismatch`、所有 level 测试都建立在它上面。那些守卫这一轮抓到了五个 bug |
| 1 轮 vs 2 轮自举 | **不能，也不需要** | GPU 路径把 `numIterations` 丢掉了（`CryptoContext.cpp:1763-1778`，只在 `devices.empty()` 的 CPU fallback 里转发）。而且 sb=59 下不需要——实测噪声在 softmax 输出上看不见 |

---

## 6. 建议的顺序

1. **先拿 `--device-memory` 的数**。上面的 `everything else 9.0 GiB` 是"calibrated from
   round 3, not derived"，而你实测 sb=50 是 25.2 GB、sb=59 是 29.3 GB——**这 4 GiB 的差
   模型解释不了**（`budget.py:key_bytes` 没有位宽参数，一个 RNS tower 不管素数 50 位还是
   59 位都是每系数一个 64 位字）。在那之前所有显存结论都带这个不确定性。
2. **融合 stage 03/04/05**（已知能拿掉 `rotated` 那 2 GiB，见 STATUS.md 2026-09-10）。
   这一条不用改 depth 就能把 headroom 抬到 4.7 GiB，factored 直接装得下。
   **比改 LayerNorm 便宜得多，建议先做这个。**
3. factored key 装下之后，每层旋转减半（8138 → 4117）——**这才是性能杠杆**。
   你那次 sb=50 "快" 的跑 GPU util 也只有 1-3%，说明是 host-bound；
   旋转次数减半直接砍掉一半的 kernel launch 和 key-switch。
4. LN 内部自举 / depth 33 作为后备：它更贵（要重推 level 调度），
   但如果第 2 条不够，它是下一个 1.8 GiB。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
