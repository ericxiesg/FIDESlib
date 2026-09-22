# conjugate 在 slot 0 误差 12-27x p50，但只有 1e-9，不是 5e5 的灾难

日期：2026-09-22。针对 `efc57a3`。**conjugate 测试跑完了。**

---

## 0. 结论

| 问题 | 答案 |
|---|---|
| conjugate 在 slot 0 是否显著大于其他槽 | **是**。slot 0 误差 = 其他槽 p50 的 **12-27 倍** |
| 但 100x 断言阈值触发了吗 | **没有**。测试 passed |
| 误差幅度 | **1e-9**。和设备上的 5e+05 差 **14 个数量级** |
| conjugate 是 07d 灾难的原因吗 | **不太可能**。slot 0 确实比其他槽大，但幅度完全在正常 CKKS 噪声范围内 |
| 第一次 he_inv 的 `low` | Run 1/3 无 warning（全在 [0.04688, 1] 内）；Run 2 low=0.08573 |

---

## 1. conjugate 测试结果

测试 `test_conjugate_at_slot_zero_its_own_fixed_point` 在 CPU 和 GPU 上各跑了一次。

### 1.1 GPU 结果（bench 参数，log_n=16）

```
conjugate(real) == real, 0 levels down (level 37):
  slot 0     1.14643e-09
  every other slot: max 9.75575e-10  p50 9.59335e-11
  slot 0 / p50 of the rest: 12.0x

real + conj(real) == 2*real, 0 levels down (level 37):
  slot 0     1.19526e-09
  every other slot: max 1.66214e-09  p50 1.06254e-10
  slot 0 / p50 of the rest: 11.2x

conjugate(complex), 0 levels down (level 37):
  slot 0     1.23512e-09
  every other slot: max 8.61695e-10  p50 9.6403e-11
  slot 0 / p50 of the rest: 12.8x
```

**9 组测量（3 种操作 × 3 个 level drop），slot 0 误差全部在 1.1e-9 ~ 1.2e-9 之间。**
其他槽的 max 在 8.6e-10 ~ 1.7e-9 之间，p50 在 ~9.6e-11。

### 1.2 slot 0 / p50 比值汇总

| 操作 | drop=0 | drop=8 | drop=16 |
|---|---|---|---|
| conjugate(real) | 12.0x | 11.5x | 12.5x |
| real + conj(real) | 11.2x | 11.1x | 11.3x |
| conjugate(complex) | 12.8x | 12.7x | 12.4x |

**slot 0 误差稳定是其他槽 p50 的 11-13 倍。** 这不是噪声——它在 9 组测量中完全一致。

### 1.3 但幅度完全不够

slot 0 的绝对误差是 **1.1e-9**。设备上 07d slot 0 的误差是 **5.8e+05**（本机 0.049）。

**差 14 个数量级。** conjugate 给 slot 0 多带了一点噪声（12x p50），但这个"多"是 1e-9 级别的。

### 1.4 关键对比：slot 0 vs 其他槽的 max

| | slot 0 | 其他槽 max | 比值 |
|---|---|---|---|
| conjugate(real), drop=0 | 1.15e-9 | 9.76e-10 | 1.17x |
| real+conj(real), drop=0 | 1.20e-9 | 1.66e-9 | **0.72x** |

**slot 0 的绝对误差和"其他槽的最大值"几乎一样。** 在 max-norm 下，slot 0 根本不突出——其他槽里也有误差接近 slot 0 的。只是 slot 0 稳定地比 p50 高 12 倍，而其他槽的 max 是偶发的尾部。

---

## 2. 第一次 he_inv 的 `low`

你问的三次跑 `[range] he_inv observed [low, high]`：

| 跑 | 第一次 he_inv | low | high |
|---|---|---|---|
| 1 | **无 warning**（全在 [0.04688, 1] 内） | — | — |
| 2 | ⚠️ `[0.08573, 1.056]` | 0.08573 | 1.056 |
| 3 | **无 warning** | — | — |

**Run 2 的 low = 0.08573**（unlifted = 0.0286，距 0 是 1.83σ）。
Run 1 和 3 的第一次 he_inv 完全通过，没有触发 range guard。

**三次里只有一次第一次 he_inv 越界**——和"100% 失败"不符。设备上 07d 的灾难不是第一次 he_inv 越界造成的，而是**第二次 he_inv**（07d 的输出传到 07e，爆到 1e+12）。

---

## 3. 总结

| 假设 | 状态 |
|---|---|
| 自举误差有位置结构 | ❌ 否掉（worst 行散布） |
| 输入形状导致不同误差 | ❌ 否掉（稀疏 vs 稠密一样） |
| slot 0 分母最小 → 最容易被推出界 | ❌ 否掉（slot 0 排 688/8448） |
| conjugate 在 slot 0 有特殊误差 | ⚠️ **slot 0 确实 12x p50，但只有 1e-9，不够解释 5e+05** |

**conjugate 不是直接原因。** 但 slot 0 稳定地 12x 其他槽的 p50 这个现象是真的——也许在 Goldschmidt 的 6 轮迭代里被放大了？1e-9 经过 6 轮 `a*(2-b*a)` 能不能变成 5e+05？

**如果你认为值得追这条线：** 可以在 `_restore_magnitude` 的每轮 conjugate-add 后单独报 slot 0 的误差，看它是否在迭代中被放大。

---

## 4. 修了测试文件

`test_he_inv_primitives.py` 缺 `import os`，已加。
