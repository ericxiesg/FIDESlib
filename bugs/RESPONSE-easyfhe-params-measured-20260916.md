# sb=59/fmb=60 实测：20.3 bit 精度，error=1.55e-6，he_inv 完全可用

日期：2026-09-16。针对 `2b29025`。

---

## 实测结果

EasyFHE 参数：`scaling_bits=59, first_mod_bits=60, q0/Delta=2`。

只测小值（q0/Delta=2，可用窗口 ~0.6）：

```
v=0.1: out=[0.10001, -0.09999, 0.05000]  max_error=1.27e-05  17.3 bits  C=2^42.7
v=0.5: out=[0.50000, -0.50000, 0.25000]  max_error=1.55e-06  20.3 bits  C=2^39.7
v=1.0: out=[1.00001, -1.00002, 0.50002]  max_error=2.40e-05  16.3 bits  C=2^43.7
```

---

## 关键对比

| 配置 | max_error | precision | error / inv_epsilon |
|---|---|---|---|
| sb=50 fmb=55 (现在) | 1.57e-02 | 11.0 bit | **3200%** |
| sb=55 fmb=60 | 4.44e-04 | 16.1 bit | **91%** |
| **sb=59 fmb=60 (EasyFHE)** | **1.55e-06** | **20.3 bit** | **0.3%** |

**v=0.5 的 20.3 bit 精度给出 error=1.55e-6，是 inv_epsilon (4.88e-4) 的 0.3%。**
Goldschmidt he_inv 完全可用。

即使最差的 v=1.0（16.3 bit, error=2.4e-5），也只占 inv_epsilon 的 4.9%。

---

## C 不是恒定的

```
v=0.1: C = 2^42.7
v=0.5: C = 2^39.7
v=1.0: C = 2^43.7
```

C 随幅度变化——在 q0/Delta=2 下，不同幅度处于 sine 近似曲线的不同位置。
v=0.5（界 2 的 25%）处于最佳区域，v=1.0（界 2 的 50%）接近边缘。

**协作者的外推（C=2^38.9, 21 bit）和实测的 v=0.5（C=2^39.7, 20.3 bit）吻合。**
但 v=1.0 的 C 更大（2^43.7），说明界边缘精度下降明显。

---

## 对两个 bug 的修法路径确认

### Bug 1 (softmax 精度): ✅ sb=59/fmb=60 解决

error=1.55e-6 << inv_epsilon=4.88e-4。he_inv 可用。

### Bug 2 (stage 15 超界): ⚠️ 必须先修

q0/Delta=2，可用窗口 ~0.6。真实 stage 15 残差 max=64.81，**必须缩放**。

协作者已实现 stage 15 缩放代码（`489fabd`）：
- `residual_scale` 参数折进 `encode_layer` 的明文权重
- `LayerNormStages.residual_scale` 调整 stage 16 的 variance bounds
- `halve` 标志从绝对值判断改为显式参数

### 正确顺序

```
1. ✅ stage 15 缩放代码已实现 (489fabd)
2. → 在 ClearEngine 上验证 stage 16 输出不变（尺度不变性精确）
3. → 迁移到 sb=59/fmb=60
4. → 跑 THOR FHE benchmark 验证端到端
```

---

## 测试

222 passed, 14 skipped（含 4 条新 stage 11 layernorm 测试，覆盖 residual_scale 的
尺度和 halve 标志行为）。
