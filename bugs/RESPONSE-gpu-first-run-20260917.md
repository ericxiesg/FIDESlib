# GPU THOR FHE 第一次端到端：sb=50 跑通但精度炸，sb=55/59 极慢

日期：2026-09-17。针对 `79ac169`。

---

## 1. sb=50（默认）跑通，19 分钟

```
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
  --refresh-after-dense --binary-rotations \
  --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16
```

```
layer 0: 1121.60s (98.6%)
total: 1137.15s (19 min)
GPU memory: 25.2 GB / 32 GB
```

**精度完全失败：**
```
hidden MAE 7.533e+17  max 4.257e+18
encrypted accuracy 0%  (label 翻转)
```

sb=50 的 bootstrap 精度 ~10 bit，22 次 bootstrap 累积后噪声完全淹没消息。

---

## 2. sb=59 跑了 4 小时未完成，kill 掉

```
scaling-bits 59, first-mod-bits 60
GPU memory: 29.3 GB / 32 GB
```

跑了 243 分钟，GPU util 0%，只有 ~1 核 CPU 在用。卡在 stage 07 softmax。
**疑似某个操作在 sb=59 下回退到 CPU 或 hang 住。**

---

## 3. sb=55 跑了 2 小时未完成，kill 掉

```
scaling-bits 55, first-mod-bits 60
GPU memory: 29.3 GB / 32 GB
GPU util: 1-4%
```

跑了 120 分钟，还在 stage 07。比 sb=50 慢 6x+。

---

## 4. 性能对比

| scaling_bits | 时间 | GPU util | 精度 | 状态 |
|---:|---|---:|---|---|
| 50 | 19 min | 1-3% | MAE=7.5e17 ❌ | 跑通但炸 |
| 55 | 2h+ (未完) | 1-4% | ? | kill |
| 59 | 4h+ (未完) | 0% | ? | kill |

**sb=50 → sb=55 慢 6x+，sb=50 → sb=59 慢 12x+。** 这个减速不线性。
sb=50 的 "快" 可能是因为 bootstrap 精度低 = 少 limb = 快。
sb>50 时 GPU util 极低（0-4%），说明大部分时间在 CPU 上。

---

## 5. 下一步

1. **`--extra-rotation-keys 6`**：binary rotations 4.5x → ~2x 旋转数，
   代价 +1GB key memory（29.3→30.3 GB），可能大幅加速
2. **调查 sb>50 时 GPU util 低的原因**：可能是某个 kernel 在大 limb 下的性能问题
3. **sb=52 折中**：比 sb=50 精度高，但可能比 sb=55 快

请定方向。
