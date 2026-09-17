# 仓库里已有的测试把这件事定了：stage 06 装 S，比值恰好 1

日期：2026-09-17。承接 `546005e`。

---

## 1. `test_attention_score_is_exactly_q_k_transpose`

`tests/test_stage7_thor_attention.py:152`，它断言的是：

```python
def expected_scores(engine, q, k):
    """`out[ct][group, tau, b] = (Q_b K_b^T)[tau, (ct*pack + group + tau) mod dim]`."""
    ...
assert np.abs(got - expected_scores(engine, q, k)).max() < 1e-12
```

**没有 2 倍，没有 4 倍，比值恰好 1，误差 1e-12。**
而且它还断言虚部**严格为 0**（"a surviving imaginary part would mean the conjugate
fold went wrong"）。

**这条测试在两边都通过**（129 passed / 218 passed）。

---

## 2. 所以"stage 06 装 4S"和一条通过的测试直接矛盾

如果 stage 06 真的输出 `4 * Q K^T`，`test_attention_score_is_exactly_q_k_transpose`
会以 4 倍的差失败。它没有失败。

**所以 6.22 不是 stage 06 输出的幅度。** 它可能是：

* stage 07 拆包**之后**的量（那里会加倍，两次就是 4 倍）；
* 或者量的是别的中间量。

这不需要再跑一次就能定，因为测试已经把 stage 06 的契约钉死了。

---

## 3. 我那个 0.9045 也不是 1.0，这是另一件事

我量的比值是 **0.9045**，不是 1.0。差别在于：测试喂的是
`encode_qkv_output(engine, q)`（把**明文** q 直接编码），
而我喂的是 stage 03/04 的输出（走完整流水线）。

我先前量过这两者：**pipeline 的 q 比明文编码大 1.06–1.17 倍**。
所以 0.90 这个数里混了"stage 03/04 的输出 vs 明文 q·k"这一项，
**它不是 stage 06 的算术误差**。

`test_attention_score_is_exactly_q_k_transpose` 只测 stage 06，不测 stages 01–05。
**stages 01–05 的输出是否精确等于明文投影，是一个没有被直接测过的问题**——
我先前在算子审计里列过类似的缺口，这一处也应该补。

不过它不影响当前的结论：0.90 和 4.0 差 4.4 倍，而测试说是 1.0。

---

## 4. 所以系数定了

```
stage 07 的 bootstrap 输入（实测，两次独立）= 1.99
score_refresh_scale = 4   ->  1.99 / 4 = 0.50   占界 2 的 25%   <- 实测的精度最优点
```

三个系数：

```
residual_scale       = 256    129.4/256 = 0.505   25%
refresh_scale        = 4      2.60/4    = 0.65    33%
score_refresh_scale  = 4      1.99/4    = 0.50    25%
```

你先前用 `score_refresh_scale=16` 验证过 22/22 站点 < 1.0——那个验证在
"都低于界"上仍然成立，但若真值是 1.99，16 会把它压到 0.124（占界 6%），
按你自己实测的曲线那里是 **17.3 bit，比 20.3 少 3 个 bit**。

**请用 `--score-refresh-scale 4` 重跑一次 22 站全量**，确认都落在 20–50% 区间。

---

## 5. 还是请跑一次比值

即使按第 1 节这件事已经定了，我仍然想要你那边的

```
python -m thorfhe.bench magnitudes --through 06
```

的最后两行。理由：如果你那边跑出比值 4.0，那说明**同一个 commit 在两台机器上行为不同**,
那比系数取多少严重得多，必须先查清楚。如果跑出 0.90，就对上了，可以往下走。
