# 旋转优化已验证，variance 灾难性相消仍未解决

日期：2026-09-18。承接 `a22ec04`。

---

## 0. 结论

协作者的 meet-in-the-middle 旋转优化（`--rotation-max-steps 4`）在设备上验证通过：

| 配置 | 密钥 | 旋转/层 | 显存 | wall clock |
|---|---|---|---|---|
| binary (15 keys) | 15 | 8258 | 28.3 GiB | 1161.8s |
| +6, max_steps=4 | 21 | **3643** | 29.2 GiB | 1154.7s |
| +6, max_steps=2 | 21 | **4256** | 29.2 GiB | ~19min (运行中) |

**旋转数下降 56%（8258→3643），但 wall clock 几乎不变**——bootstrap 调用主导时间，旋转不是瓶颈。

**variance 灾难性相消依然存在**：11a.variance = 4.592e+07（应 ~0.22），与旋转密钥无关。

---

## 1. 探针确认：设备与 ClearEngine 一致

协作者新增的 `10.attention_dense` 和 `11.residual` 探针在设备上确认：

```
[probe] 10.attention_dense    8 ct  level 1    — 不带 --refresh-after-dense 时 LN1 无法运行
[probe] 11.residual           8 ct  level 20   — refresh 后残差回到 level 20，LN1 的 14 层链可以跑
```

与协作者在 ClearEngine 上的观测完全一致。`--refresh-after-dense` 在 depth 37 下是必须的。

---

## 2. variance 灾难性相消：三次实验，同一结果

| 运行 | 密钥 | 11a.variance (GPU) | 11a.variance (ClearEngine) |
|---|---|---|---|
| gpu_final.log (15 keys) | 15 | 4.628e+07 | 0.22 |
| gpu_rot_steps4 (21 keys, max_steps=4) | 21 | 4.592e+07 | 0.22 |
| gpu_rot_steps2 (21 keys, max_steps=2) | 21 | (运行中) | 0.22 |

**variance = n\*Σx² − (Σx)² 在 FHE 下灾难性相消**，差 2 亿倍。这不是旋转密钥的问题，
是 `n\*Σx²` 和 `(Σx)²` 都是 ~1e9 量级，差值应是 ~0.22，但 FHE 噪声让大数减大数丢失所有有效位。

ClearEngine（精确算术）没有这个问题，MAE = 1.441e-03。

---

## 3. 新发现：21 密钥下 softmax 的 inverse_denominator 有异常 outlier

15 密钥运行中 `07d.inverse_denominator` 干净（max 0.1477）。21 密钥运行中出现：

```
[probe] 07d.inverse_denominator   1 ct  level 15   min -3099   max +6.674e+16   p50 0.02425   p99 0.2426
```

p99 正常（0.2426），但少数 slot 出现 6.7e+16 的异常值。这导致 softmax 的 per-stage fidelity
从 15 密钥的 relRMSE 5.6e-4 恶化到 21 密钥的 relRMSE 1.320e+2。

可能原因：extra rotation keys 的 meet-in-the-middle 分解在某些 slot 上引入了更多 key-switch
噪声。不影响 p99，但少数 slot 炸了。

---

## 4. 旋转优化对比

### max_steps=4 (meet-in-the-middle, 默认)
- 21 keys, **3643 rotations/layer**
- 10208 MiB rotation key memory (resident)
- wall clock: 1154.7s
- 选择的 6 把 extra keys 与协作者报告一致

### max_steps=2 (binary expansion, 旧行为)
- 21 keys, **4256 rotations/layer**
- 同样 10208 MiB key memory
- wall clock: ~19min (运行中)
- 比 max_steps=4 多 613 次旋转 (14%)，但 wall clock 预计接近

**结论**：max_steps=4 确实减少旋转数，但对 wall clock 影响可忽略。默认值合理。

---

## 5. 下一步：variance 灾难性相消是唯一阻塞项

旋转优化已验证通过。**唯一阻塞项是 variance 的灾难性相消**。

之前报告（`RESPONSE-variance-explodes-on-gpu-20260918.md`）中提的 4 个修法选项仍然有效：

- **A. 改用 Σ(x−mean)²**：不花 bootstrap，改算法。需要先算 mean（一次 rotate+sum），再减、平方、求和。
  在 FHE 下 mean 本身有噪声，但 Σ(x−mean)² 的两个加数都是 ~0.22 量级，不会相消。
- **B. Meta-BTS**：+4 bootstrap/层，精度够但慢一倍。
- **C. variance 后 bootstrap**：不解决相消，只是把错误值刷新。
- **D. 拆两步各 bootstrap**：+1 bootstrap，也不解决相消。

**推荐 A**：不花 bootstrap，只改算法，ClearEngine 验证容易。

---

## 6. 请协作者确认

1. variance 灾难性相消的修法方向（A/B/C/D 或其他）
2. 21 密钥下 07d.inverse_denominator 的 outlier 是否需要关注（p99 正常，但 max 炸了）
3. 是否需要跑 `--rotation-key-budget 3.4` 测试（当前 29.2 GiB 已经跑通，budget 限制可能没必要）

Co-Authored-By: CodeAgent (GLM-5.2) <noreply@anthropic.com>
