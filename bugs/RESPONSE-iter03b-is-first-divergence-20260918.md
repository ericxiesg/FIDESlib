# he_inv 迭代探针：iter03_b 是第一个发散步

日期：2026-09-18。承接 `ad46a10`。

---

## 0. 结论

**`THORFHE_DEBUG=1` 逐轮探针定位：iter03_b 是第一个发散步。**

iter01 和 iter02 在 GPU 上和 ClearEngine 一致。iter03_b 的 max 从 0.004 变成 219.6（差 5.8 万倍），
p99 仍然是 0.01（和 ClearEngine 的 0.004 接近）。**少数 slot 在第三轮开始发散，之后每轮平方，爆炸。**

---

## 1. 第一组 he_inv 逐轮数据（softmax inverse_denominator）

| 步骤 | GPU max | GPU p99 | ClearEngine max | ClearEngine p99 | 判断 |
|---|---|---|---|---|---|
| iter01_a | 1.03 | 0.996 | 0.981 | 0.980 | ✅ |
| iter01_b | 1.442 | 0.153 | 0.214 | 0.120 | ✅ |
| iter02_a | 0.296 | 0.253 | 0.229 | 0.227 | ✅ |
| iter02_b | 0.179 | 0.019 | 0.016 | 0.016 | ✅ |
| iter03_a | 0.238 | 0.131 | 0.138 | 0.134 | ✅ |
| **iter03_b** | **219.6** | **0.010** | 0.00376 | 0.00376 | ❌ **首发** |
| iter04_a | 6.52e+06 | 0.169 | 0.141 | 0.130 | 💥 |
| iter04_b | 2.0e+07 | 0.071 | 0.0039 | 0.0039 | 💥 |
| iter05_a | 3.59e+16 | 0.199 | 0.167 | 0.159 | 💥 |
| iter05_b | 1.1e+17 | 1.664 | 0.0039 | 0.0039 | 💥 |

**iter03_b 的 p99 = 0.010**（和 ClearEngine 的 0.00376 同量级），**但 max = 219.6**。
是**少数 slot** 在第三轮 Newton 迭代中发散，不是整体噪声。

---

## 2. 发散机制

Newton 迭代 `y ← y·(2 − x·y)` 在 `x ≈ 0` 时：
- `y_{n+1} ≈ 2·y_n`（每轮翻倍）
- 7 轮后 `y ≈ 128`

07c.denominator 的 p50 = 1.012e-11——**大部分 slot 是 padding（≈0）**。
这些 slot 的 Newton 迭代每轮翻倍，iter03 时 `b ≈ 2³ = 8`，但实测 219.6 更大，
说明 `_restore_magnitude` 的整数重缩放可能放大了。

iter04 时 `a = a·(2 − k·b)`，`b = 219.6` 让 `a` 变成 ~6.5e6。
之后每轮 `a²` 增长，iter05 变 3.6e16——和 07d 的 5.3e16 一致。

---

## 3. 为什么协作者的空槽模型没预测到

协作者在 `ad46a10` §2 里测了空槽泄漏，结论是"饱和在 1，做不出 5.3e16"。
但那个模型假设 `a` 和 `b` **同等泄漏**，比值收敛到 1。

实际 GPU 上 `a` 和 `b` 的噪声可能不对称：
- `b` 是 `denominator` 的密文本身（直接 bootstrap 过）
- `a` 是 `align(ones, bootstrap(denominator))`——多了一次 `align` 和 `bootstrap`
- 如果 `a` 在空槽上的泄漏比 `b` 大，比值不收敛到 1，而是持续增长

---

## 4. 15-key 对照：07d outlier 不是密钥问题

15-key 跑（`gpu_15keys_probes.log`）同样出现 07d outlier：

```
15-key (新代码):  07d.inverse_denominator  max 3.384e+16  p99 0.1917
21-key (新代码):  07d.inverse_denominator  max 5.298e+16  p99 0.2381
15-key (旧代码):  07d.inverse_denominator  max 0.1477     p99 0.1092  ← 干净
```

**旧代码（35841d9）15-key 下 07d 是干净的。** 新代码（a22ec04 起）15-key 也炸了。
不是密钥的问题，是代码变更引入的。`numeric.py` 只加了探针没改计算，
所以嫌疑在 `a22ec04` 对 `stages.py`（4 行）或 `dense.py`（1 行）的修改。

---

## 5. 修法方向

### 短期：mask 掉空槽
在 `he_inv` 迭代前，把 padding slot 的值显式设为 1（或任何非零值），
迭代结束后再 mask 回 0。这样 Newton 迭代在空槽上 `x = 1, y → 1`，不会发散。

### 中期：改用有界迭代
用 `y ← y·(2 − x·y) · mask(x > threshold)` 或在迭代中 clamp `y` 到 [0, 2]。

### 长期：避免空槽
重新设计 schedule 让 padding slot 的值不为零（比如填 1）。

---

## 6. --time-ops 结果摘要

```
encode_to_light_plaintext    19,137 calls    1109.65 s    96.2%    58 ms/call
bootstrap                        22 calls       4.16 s     0.4%   189 ms/call
rotate                        3,643 calls       0.77 s     0.1%   0.2 ms/call
```

协作者已推 `7a03a0d`（从 2 个 RNS tower 编码 light plaintext）修这个。

---

## 7. 请协作者确认

1. **iter03_b 是首发步**——是否同意根因是空槽 Newton 发散？
2. **15-key 旧代码 07d 干净、新代码 07d 炸**——能否 bisect `a22ec04` 里 `stages.py` 或 `dense.py` 的改动？
3. **mask 空槽**是否可行？在 `he_inv` 前 mask padding slot 为 1，迭代后 mask 回 0

Co-Authored-By: CodeAgent (GLM-5.2) <noreply@anthropic.com>
