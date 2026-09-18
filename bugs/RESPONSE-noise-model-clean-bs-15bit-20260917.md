# noise model 干净 + bootstrap 实测 15.2 bit：GPU 2.4e19 不是 bootstrap 精度问题

日期：2026-09-17。针对 `13ff856`。

---

## 1. ClearEngine + noise model = 干净

```
python -m thorfhe.bench fhe --engine clear --layers 1 --limit 1 \
  --refresh-after-dense --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --noise-model --bootstrap-noise-only \
  --scaling-bits 59 --first-mod-bits 60 --bootstrap-precision-bits 20 --lenient
```

```
hidden MAE 7.477e-02  max 8.344e-01
label agreement 100%
```

**和无噪声完全一致。** bootstrap 误差在 sb=59 的 noise model 下不导致精度问题。

---

## 2. 单次 bootstrap 实测精度：15.2 bit

sb=59 / fmb=60 / depth=37 / (3,3) / SPARSE，GPU 上直接测：

```
 input   bootstraped   abs_err    bits
 0.5000    0.499995    4.77e-06    17.7
 1.0000    1.000002    2.43e-06    18.7
 2.0000    1.999997    3.48e-06    18.1
-1.0000   -0.999990    1.02e-05    16.6
 0.1000    0.100023    2.30e-05    15.4
 3.1400    3.140026    2.60e-05    15.2
-2.7100   -2.710001    6.42e-07    20.6
 0.0010    0.001020    1.96e-05    15.6

max abs error: 2.60e-05
bits (from max): 15.2
median abs error: 7.47e-06
```

**15.2 bit，不是 20.3 bit。** 比上一轮的测量低 5 bit。差异可能来自：
- 上一轮测的是不同 depth/level_budget
- 这一轮是 depth=37 / (3,3)，匹配生产配置
- bootstrap 输入 level 不同（生产是链底，上一轮可能在高 level）

---

## 3. 15 bit 不解释 2.4e19

22 次 bootstrap 累积 15 bit 误差：
- 每次 ~2.6e-5 绝对误差
- 22 次最坏情况累积：22 * 2.6e-5 = 5.7e-4
- 即使有放大效应（乘法、softmax），也不会到 1e19

**GPU 上的 MAE=2.4e19 不是 bootstrap 精度问题。** noise model 覆盖了 bootstrap 误差
并且不炸——说明病因在 noise model 没有覆盖到的地方。

---

## 4. 可能的非 bootstrap 精度问题

| 来源 | noise model 是否覆盖 | 在 ClearEngine 是否精确 |
|---|---|---|
| bootstrap 误差 | ✅ 覆盖 | ✅ 精确 |
| rescale 舍入 | ❌ 未覆盖 | ✅ 精确 |
| key-switch 噪声 | ❌ 未覆盖（`--bootstrap-noise-only` 跳过） | ✅ 精确 |
| truncated key 误差 | ❌ 未覆盖 | ✅ 精确 |
| 多项式近似 | ❌ 未覆盖（he_exp/he_inv 的 Chebyshev） | ❌ 也有（但量级小） |

**最可能的 culprit：key-switch 噪声 + truncated keys。**

binary rotations 用 15 个 key 替代 210 个，每次旋转做 key-switch，
truncated key 在低 level 下可能引入额外噪声。
一层有 8138 次旋转 = 8138 次 key-switch，累积噪声可能远超 bootstrap。

协作者说 key-switch 累积 8.2e-11（vs bootstrap 4.9e-4），但那是在 ClearEngine
模型里——**GPU 上的 truncated key 可能比模型差得多。**

---

## 5. 下一步

1. **跑 `--per-stage`** 定位 GPU 上哪个 stage 开始发散
2. **不带 `--bootstrap-noise-only`** 跑 noise model（加上 key-switch 噪声），
   看是否复现 2.4e19——但协作者说这会 OOM（每操作两个 32768 数组）
3. **`--lazy-weights`** 还没在远端代码里，需要等协作者推送
