# StC gain 不是 sqrt(N) 固有的；抬 Delta 在 32GB GPU 上跑不起来

日期：2026-09-16。针对 `aa84196`。

---

## 实验 A：sqrt(N) 增益

如果 StC 的增益是 DFT 类变换对非相干噪声的固有 sqrt(N) 放大，
换 log_n 应该让增益按 sqrt(N) 变化。

```
log_n=14  N=16384  slots=8192   sqrt(N)=128   sqrt(slots)=90.5
  st3_p50=8.12e-07  st4_p50=1.72e-04  gain=211.9

log_n=15  N=32768  slots=16384  sqrt(N)=181   sqrt(slots)=128.0
  st3_p50=1.38e-06  st4_p50=2.94e-04  gain=213.4

log_n=16  N=65536  slots=32768  sqrt(N)=256   sqrt(slots)=181.0
  st3_p50=8.40e-06  st4_p50=2.62e-03  gain=311.9
```

### 分析

**增益不是 sqrt(N)。**

| log_n | sqrt(N) | sqrt(slots) | 实测 gain |
|---|---|---|---|
| 14 | 128 | 90.5 | **212** |
| 15 | 181 | 128 | **213** |
| 16 | 256 | 181 | **312** |

- log_n=14→15：N 翻倍，sqrt(N) 从 128→181（+41%），gain 从 212→213（+0.5%）
- log_n=15→16：N 翻倍，sqrt(N) 从 181→256（+41%），gain 从 213→312（+47%）

**增益不随 sqrt(N) 变化。** 协作者的"固有 sqrt(N) 放大"假设不成立。
StC 的增益远大于 sqrt(N)（212 vs 90.5 at log_n=14），且增长模式不匹配。

### st3_p50 的增长值得注意

```
log_n=14: st3_p50 = 8.12e-07
log_n=15: st3_p50 = 1.38e-06  (1.7x)
log_n=16: st3_p50 = 8.40e-06  (6.1x)
```

Stage 3（模约简）的精度也随 log_n 变差。这和 sqrt(N) 一致
（sqrt(16384)/sqrt(32768) = 0.71，sqrt(32768)/sqrt(65536) = 0.71，
但实测是 1.7x 和 6.1x），也可能和 Chebyshev 系数在不同 N 下的选择有关。

---

## 实验 B：抬 scaling_bits（Delta 2^50 → 2^52, 2^54）

协作者建议测"单独抬 Delta"看绝对误差是否按 1/Delta 下降。

```
sb=50 depth=37 q0/Delta=32: max_error=1.58e-02  precision=11.0 bits
sb=52 depth=35: OOM (GPU 32GB 不够)
sb=54 depth=33: OOM
```

**抬 scaling_bits 在 32GB GV100 上跑不起来。** 更大的 Delta 意味着更多 RNS limbs，
即使降低 depth 也无法在 32GB 显存内完成 bootstrap。

### 但结论可以从已有数据推出

之前 first_mod_bits 实验（55→57，q0/Delta 32→128）显示**绝对误差不变**（0.022→0.022）。
如果误差是 `C/Delta` 的形式（C 是某个常数），那么：
- 改 q0（first_mod_bits）不改 Delta，不改 1/Delta → 误差不变 ✓
- 改 Delta（scaling_bits）应该让误差按 1/Delta 变 → 但我们测不了

**但 first_mod_bits 实验已经证明误差不随 q0/Delta 缩放。** 如果误差 = C/Delta，
那 C = error × Delta = 0.016 × 2^50 = 2^45.7。这个 C 是系数域的某个量。

协作者的 §4 假设是 `constantEvalMult` 缩小 2^16+，Chebyshev 在缩小尺度上做，
舍入误差 ~1/Delta，最后 `corFactor` 放大回去。如果这个模型对，
那误差 = (1/Delta) × corFactor = (1/2^50) × 4 = 2^-48 ≈ 3.5e-15。
但实测是 0.016 = 2^-6。**差了 2^42，远超 corFactor=4 能解释的范围。**

所以误差不是简单的"舍入误差被 corFactor 放大"。有别的放大机制。

---

## 修正协作者 §1 的解读

### corFactor = 4

`whole_bootstrap max 6.31e-02 / stage4 max 1.58e-02 = 3.99`。
确认 `corFactor = 4`，`correction = 2`。uint32_t 回绕没有发生。

### 非零 stage 4 的 max 0.75 是信号

`3.0 / 4 = 0.75`。是消息被恢复但还没乘 corFactor。p50 (2.63e-03) 和零密文 (2.61e-03)
一样，确认底噪与消息无关。

### Stage 3 精度 ~22 bit

`p50 = 8.40e-06, q0/Delta = 32`。`8.40e-06 / 32 = 2.6e-7 = 2^-21.8`。
约 22 bit，是健康的 CKKS 模约简精度。正弦那一步没问题。

---

## 当前状态

| 假设 | 状态 |
|---|---|
| StC gain 是固有 sqrt(N) | ❌ 排除（gain 212-312, 远大于 sqrt(N) 90-256, 增长模式不匹配） |
| 抬 Delta 能降低误差 | ⚠️ 无法在 32GB GPU 上测试 |
| 误差 = 舍入 × corFactor | ❌ 不成立（预测 3.5e-15, 实测 0.016, 差 2^42） |
| corFactor 有 uint32 回绕 | ❌ 排除（corFactor = 4, 正常） |
| Stage 3 (sine) 精度不够 | ❌ 排除（22 bit, 健康） |

**StC 的 212-312x 增益不是 DFT 固有的，是某个额外放大机制。** 下一步应该往 StC 内部查。

---

## 测试

218 passed, 13 skipped（含 8 条新 softmax 算子测试）。
