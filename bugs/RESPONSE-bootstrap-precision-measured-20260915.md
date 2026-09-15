# 实测：bootstrap 精度、NoiseLevel、softmax 分母——三个数全部到位

日期：2026-09-15。针对 `147e130` 和 `915793e`。

---

## 1. 三个数

### 1.1 bootstrap 实测绝对误差

```
Params: log_n=16, depth=37, dnum=4, scaling_bits=50, first_mod_bits=55
        SPARSE_TERNARY, bootstrap_level_budget=(3,3)

Fresh:     level=37 noise=1
Bootstrap: level=20 noise=1  (after Engine.bootstrap's rescale)
Raw EvalBootstrap: level=21 noise=2

Values:    expected = [2.0,      -3.0,       0.5      ]
           got       = [2.004673, -2.998887,  0.483042]
           abs_error = [4.67e-3,   1.11e-3,   1.70e-2 ]
           max_abs_error = 1.70e-2

q0/Delta = 2^55/2^50 = 32
bootstrap precision = -log2(0.017 / 32) = 10.9 bits
```

**10.9 bit，不是 20-25 bit。差了 10-14 bit。** 容差已改回 1e-3，精度测试标 xfail。

### 1.2 `07c.denominator` 的 p50（THOR FHE 运行，layer 0, sample 0）

```
[probe] 07c.denominator  |x| p50 7.026e-05  p99 3.357e-04  max 5.843e-04
inv_epsilon (layer 0) = 2^-11 = 4.883e-04
```

**p50 = 7.03e-05，是 inv_epsilon 的 0.14 倍。远低于下界。** 假设成立。

### 1.3 `07a.refreshed_scores` 的 min/max vs NARROW 窗口

```
[probe] 07a.refreshed_scores  min -10.34  max +12.43  |x| p50 0.927  p99 6.524
NARROW softmax window:  min_x = -27.25   max_x = +21.73
```

**score 范围 [-10.34, 12.43] 只占了 NARROW 窗口 [-27.25, 21.73] 的中间一小段。**
exp(12.43) ≈ 2.5e5，exp(-10.34) ≈ 3.2e-5。经过 softmax 的 1/512 缩放后，
exp 值在 [6.2e-8, 0.49] 范围，分母（128 个 exp 的和）的中位数只有 7e-5。

### 1.4 `07b.exp` 的 p50

```
[probe] 07b.exp  |x| p50 5.128e-13  p99 6.538e-06  max 2.302e-04
```

exp 的 p50 是 5e-13，极小。这和分母 p50 = 7e-5 一致：大多数槽的 exp 值都是零头。

---

## 2. NoiseLevel before `multIntScalar`（ApproxModEval.cu:80）

```
Raw EvalBootstrap: level=21, noise=2
approxModReduction 末尾的 rescale (line 82) 把 NoiseLevel 减 1
=> 进入 rescale 前 NoiseLevel = 3
=> multIntScalar 之前 NoiseLevel = 3  （multIntScalar 不改 NoiseLevel）
```

**是 3，不是 2。** 这同时解释了 metadata（NoiseLevel=2 输出）和精度（10.9 bit）两件事。

我们的配置 `N/2 == slots`（65536/2 == 32768），所以走的是 `approxModReduction`（非 sparse 路径，
`Bootstrap.cu:127`），它末尾有一次 FIXEDMANUAL rescale。如果走 `approxModReductionSparse`
（`Bootstrap.cu:133`），末尾没有 rescale，NoiseLevel 会是 3 而不是 2。

---

## 3. 因果链

```
bootstrap 内部 Chebyshev/double-angle 缺一次 rescale
  → NoiseLevel 进入 approxModReduction 末尾时是 3 而非 2
  → 末尾 rescale 后是 2 而非 1
  → 精度只有 10.9 bit 而非 20-25 bit（同一个 bug 的两面）
  → Engine.bootstrap 补一次 rescale 纠正 NoiseLevel，但精度已经丢了
  → he_inv bootstrap 分母（2e-4 量级）时，0.017 的绝对误差是分母的 85 倍
  → Goldschmidt 收到的分母完全是噪声
  → 07d.inverse_denominator p50 = 1.9e+159（数据槽发散，不只是填充槽）
  → softmax 输出是垃圾
  → 准确率 50%（随机）
```

**根因不在 he_inv，不在 softmax 标定，在 bootstrap 本身的精度。**
标定问题（score 只占窗口中间）是第二层：它让分母更小，让 bootstrap 精度问题更容易触发。
但即使标定完美，10.9 bit 的 bootstrap 在 2e-4 的分母上也会崩。

---

## 4. dnum=4 vs dnum=3

已改回 dnum=4（和 bench.py 默认一致）。dnum=4 下引擎正常构建，bootstrap 正常运行。
之前改成 3 是为了避一个建不起来的问题，但 dnum=4 下没有这个问题。

---

## 5. 测试修复

1. `stages.py`: `basis is not True` → `isinstance(basis, bool)`（已在上轮 commit）
2. `test_bootstrap_noise_level.py`: 
   - `bench_params()` 补上 `bootstrap_level_budget=(3,3)`, `rotation_indexes`, `truncate_keys`, `allow_key_grow`
   - `dnum=4`（和 bench 一致）
   - 容差改回 `1e-3`，精度测试标 `xfail`（实测 0.017 > 1e-3）
   - `test_eval_bootstrap_itself_returns_a_canonical_ciphertext` 标 `xfail`（已知 noise=2 bug）
3. `clear.py`: float scalar multiply 在 degree-2 上合法（`multScalar` 缩放全部三个分量），
   只拒绝 plaintext multiply on degree-2（`multPt` asserts `NoiseLevel < 2`）
4. `test_clear_engine_contract.py`: `test_float_scalar_multiply_refuses...` 改为测 plaintext
   路径（`multPt`），因为 float scalar（`multScalar`）在 degree-2 上是合法的

207 passed, 7 skipped（GPU 测试在 bench_params 下：1 passed, 2 xfailed）。

---

## 6. 下一步建议

1. **查 Chebyshev 系列内部的 rescale 缺失**：在 `evalChebyshevSeries` 和
   `applyDoubleAngleIterations` 里加 `PRINT` 追踪 NoiseLevel，找到它从 1 变成 3 的那一步。
2. **如果 level budget (3,3) 太小**：尝试 (4,4) 或 (5,2) 看精度是否改善。代价是 depth。
3. **标定是第二层问题**：即使 bootstrap 精度修好，score 范围 [-10, 12] vs 窗口 [-27, 22]
   仍然偏小。可能需要用 `--calibrate` 或调整 `attention_key_scale`。
