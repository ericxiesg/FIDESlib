# stage 11 一直没有探针——`layer.attention` 是唯一挂了 probe 的 owner

日期：2026-09-18。承接 `b682407`（noise_level 排除）。

---

## 0. 结论

`bench.py:472` 原来是：

```python
layer.attention.probe = magnitude_probe if trace is not None else None
```

**只挂在 attention 上。** 所以 `07a`–`07d` 能打，是因为它们是全层**唯一打得出来的探针**；
`LayerNormStages`、`DenseStages`、`FeedForwardStages` 里加任何 `self.probed(...)` 都是死代码。

`norm_1` 一直不说话，有一部分原因就是它没有嘴。已改成四个 owner 都挂。

---

## 1. stage 11 的探针和 ClearEngine 基线

新增 `11a.variance`（进 `he_invsqrt` 之前）和 `11b.inverse_sqrt`（之后）。
真实 checkpoint，`--compact --per-stage`：

```
[probe] 11a.variance        1 ct  level 17  min +0  max +0.2161  p99 0.1578
[probe] 11b.inverse_sqrt    1 ct  level 8   min +0  max +3.953   p99 3.8
[probe] 11a.variance        1 ct  level 14  min +0  max +0.1763  p99 0.1093   <- stage 16
[probe] 11b.inverse_sqrt    1 ct  level 3   min +0  max +5.68    p99 4.316
```

variant 1 的窗口是 `[min_var/max_var, 1] = [0.015, 1]`，**0.216 在里面**。
`min = 0` 是因为只有每个 token 的 slot 0 携带统计量，其余是空的——
`_check_invsqrt_range` 只看 mask 标的那些 slot，所以不受影响。

**请在设备上跑 `--per-stage` 并把这四行报回来。** 和上面对不上就定位到了。

---

## 2. 又排除两项

| 嫌疑 | 结论 |
|---|---|
| **截断的旋转密钥** | **排除**。`--allow-key-grow` 的 help 写着：不加它时，用超出计划 level 的截断密钥是**抛异常**而不是算错。你那次是数值发散没抛，所以密钥 level 是对的 |
| `EvalNegate` | **排除**。`Ciphertext::negate()` 用整数 `q_i - 1` 而不是 `multScalar(-1.0)`，注释里写明了为什么——后者会 `NoiseLevel += 1` 并把密文留在 `Delta^2`。这条已经是对的 |

`subtract(ndarray, ct)` 分支逐行看下来是干净的：明文编码用的表达式和 `multiply(ct, ndarray)`
**完全一样**（`depth - level`），而后者每个 stage 都在用。加上你在 level 37 实测 1e-13，
这条越来越不像原因。

### 现在的排除表

```
✅ noise_level 变负        （你的 ClearEngine 追踪，始终为 1）
✅ bootstrap 精度          （15.2 bit，noise model 不炸）
✅ softmax 链              （07a-07d 正常）
✅ key-switch 噪声          （1/5941767）
✅ variance 灾难性相消      （条件数 1.01）
✅ 截断旋转密钥            （超界会抛，你那次没抛）
✅ EvalNegate              （已用整数实现）
❓ subtract(ndarray,ct) 低 level  （代码干净，但 GPU 上没测过）
❓ he_invsqrt 的 GPU 累积舍入
```

---

## 3. 顺带：`--compact` 在真实 checkpoint 上是精确的

```
              hidden after layer 0        relRMSE
不开 compact   MAE 1.441e-03              2.776e-03
开 compact     MAE 1.441e-03              2.776e-03
```

逐位相同。`--compact` = 流式 QKV copy + 惰性权重编码，省 1.07 GiB。

`plan_rotations` 现在也收 `compact`——`factored_basis` 按**实测频次**挑额外密钥，
而流式让 copy 的旋转发生三次而不是一次。实测在 6 个额外密钥下贪心挑的是**同样的 21 把**，
旋转 4117 → 4237，所以现在不咬人；但计划不该按不会跑的配置来算。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
