# PADDING_FLOOR 修复未阻止发散：07d 仍 3.77e+16，11b 仍 1.81e+124

日期：2026-09-18。针对 `ae3d9b0`。

---

## 0. 结论

`PADDING_FLOOR = 0.5` 修复在设备上**未阻止发散**。使用 `--plaintext-cache` 热跑（89s），07d 和 11b 与修复前几乎完全一致：

| 探针 | 修复前 max | 修复后 max |
|------|-----------|-----------|
| 07c.denominator | 0.298 | 0.348 |
| 07d.inverse_denominator | 5.926e+16 | **3.774e+16** |
| 11b.inverse_sqrt | 2.001e+124 | **1.806e+124** |

---

## 1. 修复确实在生效

确认 `support=self._support(attention_mask)` 被传入 `he_inv`，且 `PADDING_FLOOR` 的 plaintext add 在 `refreshed` 上执行了。`he_invsqrt` 也加了同样的 lift。

但发散的 slot 可能不是 padding slot——**可能是 active slot**。协作者在 commit message 中说：

> "A live slot whose denominator is near the bootstrap's own error is still wrong by a lot - at 0.017 the smallest denominator here, 0.05, comes back 58% off - and no filler can help that."

这表示修复本身也预期不能解决 active slot 的问题。如果 iter03_b 发散的 slot 是 active slot（denominator 接近 epsilon=2^-11≈0.00049），那 PADDING_FLOOR 无法触及它。

---

## 2. 需要确认：哪些 slot 在发散？

之前 THORFHE_DEBUG=1 的逐轮探针显示 iter03_b 的 p50=0.01024、p99=0.01024、max=241.8。p50 和 p99 都很小，说明**只有极少数 slot 在发散**。

下一步应该：
1. 在 iter03_b 之后 decrypt 并打印哪些 slot 超过 1.0——是 padding slot 还是 active slot？
2. 如果是 active slot，那问题是 bootstrap 精度不够（10.9 bits），而不是 padding 的符号问题。
3. 如果是 padding slot，那 PADDING_FLOOR 的 lift 没有到达它们——可能是 `_support` 返回的 mask 和实际 padding 不匹配。

---

## 3. 缓存验证附带确认

热跑 89.39s，`plaintext cache: 8 fields loaded, 0 encoded`，query/value/scores relRMSE 与冷跑一致。缓存工作正常。

Co-Authored-By: CodeAgent <noreply@anthropic.com>
