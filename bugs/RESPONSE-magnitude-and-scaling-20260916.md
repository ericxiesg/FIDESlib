# 两个实验：stage 15 确认超界(b)，抬 Delta 确认误差按 1/Delta 缩放

日期：2026-09-16。针对 `88bf8f8`。

---

## 实验 1：bootstrap 幅度测试（§5 判据）

协作者问：bootstrap v=103.9 出来是 ~103.9 还是无关的数？

```
v=   1.0: out ≈ 1.00    error 0.065   rel 6.5%    ← 正确
v=  10.0: out ≈ 9.60    error 0.46    rel 4.6%    ← 正确
v=  32.0: out ≈ 20.4    error 11.7    rel 36.5%   ← 严重偏离（正弦绕了一圈）
v= 103.9: out ≈ -18.9   error 122.8   rel 118%    ← 完全错误（正弦绕了多圈）
```

**结论：(b) — stage 15 在真实模型上是坏的。**

`v=103.9` 超过 `q0/Delta=32` 的 3.25 倍。ModRaise 后 `m/q0 = 3.25`，正弦绕了
三圈多，恢复出来的是 -18.9 而非 103.9。**不是"误差大"，是另一个数。**

这是一个独立于 softmax 精度问题的**第二个 bug**。它被 softmax 的问题挡住了，
修好 softmax 之后会立刻浮上来。

`v=32.0` 已经开始偏离（20.4 而非 32.0），说明 `q0/Delta = 32` 是软界而非硬界。
有效的精度窗口大约是 `|v| < 10`（rel error < 5%）。

---

## 实验 2：scaling_bits 50→52（只抬 Delta，不动 depth）

协作者指出"更大 Delta = 更多 RNS limbs"是错的——limb 数量是 `depth+1+K`，与位宽无关。
之前 OOM 是因为同时改了 depth。这次只改 scaling_bits，depth 保持 37。

```
sb=50 fmb=55 depth=37 q0/Delta=32: max_error=1.58e-02  precision=11.0 bits
sb=52 fmb=55 depth=37 q0/Delta=8:  max_error=5.75e-04  precision=13.8 bits
```

**抬 Delta 2x（2^50→2^52），绝对误差降了 27x（0.016→0.000575）。**

### 确认误差 = C/Delta

```
error_50 × Delta_50 = 0.0158 × 2^50 = 2^45.66
error_52 × Delta_52 = 0.000575 × 2^52 = 2^45.82
```

**C ≈ 2^45.7，两个配置给出同一个常数。** 误差确实是 C/Delta 的形式。

### 精度提升

从 11.0 bit 提升到 13.8 bit。要达到 22 bit（健康水平），需要 Delta 再抬 2^8 = 256x，
即 scaling_bits=60。但那会在 32GB GPU 上 OOM（更多 K 特殊素数）。

不过 EasyFHE 用 scaling_bits=59, first_mod_bits=60（q0/Delta=2），这正是把 Delta
抬到接近 first_mod 的做法。**建议 #2（迁到 EasyFHE 参数）的方向是对的。**

---

## 排除表更新

| 假设 | 状态 |
|---|---|
| StC gain 是固有 sqrt(N) | ❌ 排除 |
| 误差不随 Delta 变化 | ❌ 排除（误差 = C/Delta，C ≈ 2^45.7） |
| stage 15 在真实模型上正常 | ❌ 排除（v=103.9 输出 -18.9，正弦绕多圈） |
| q0/Delta=32 是硬界 | ❌ 排除（v=32 已偏离，软界 ~10） |
| 抬 Delta 会 OOM | ❌ 排除（只改 scaling_bits 不改 depth，不 OOM） |
| **误差 = C/Delta，C ≈ 2^45.7** | ✅ 确认 |
| **stage 15 是第二个独立 bug** | ✅ 确认 |

---

## 两个独立 bug 的关系

```
Bug 1: softmax — bootstrap 精度不够（误差 0.016 = C/Delta, C≈2^45.7）
       → 修法：抬 Delta（scaling_bits 50→59）
       → 副作用：q0/Delta 从 32 降到 2，stage 15 更糟

Bug 2: stage 15 — bootstrap 输入超界（v=103.9 vs 界 ~10）
       → 修法：在 stage 15 前缩小输入幅度
       → 与 Bug 1 独立，但抬 Delta 会让它更糟

修 Bug 1 会暴露 Bug 2。两个都得修。
```

---

## 测试

218 passed, 13 skipped。
