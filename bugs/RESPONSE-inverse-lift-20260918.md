# `--inverse-lift`：把分母抬进量程上部，**零 level 代价**

日期：2026-09-18。承接 `c1e867d`。

---

## 0. 结论

上一份报告里我说"把分母抬进量程上部要花一层 level，所以先不做"。**那句话是错的。**
整数标量乘法是 **level-free** 的（`EvalMultByInteger`，`_restore_magnitude` 一直在用这个）。
所以这个改动不花 level。

`he_inv` 现在在**自举之前**把分母乘一个整数，再从返回的 `delta` 里除回去。

```
--inverse-lift 3
```

### 真实 checkpoint，layer 0，ClearEngine（**没有任何噪声模型**）

| | lift 1 | lift 3 |
|---|---|---|
| hidden MAE | 1.441e-03 | **4.325e-04** |
| hidden relRMSE | 2.776e-03 | **8.307e-04** |
| best-fit scale | 1.9997 | **1.9999** |
| logits relRMSE | 3.849e-05 | **4.624e-06** |
| probabilities L1 | 1.409e-06 | **1.783e-07** |

**每一项都好 3–8 倍**，而且 best-fit scale 更靠近理想的 2.0——所以不是"换了个缩放"，是真的更准。

原因：Goldschmidt 的迭代表是按 `epsilon` 推的，而 softmax 的分母实际落在量程**底部**。
抬上去就是把量程用满。

### 注入设备实测的 0.017 自举误差之后

| lift | 无噪声 | 0.017 噪声 |
|---|---|---|
| 1 | 4.1e-07 | **35.58** |
| 2 | 4.3e-05 | 1.35 |
| 3 | 3.1e-06 | **0.6256** |
| 4 | **ValueError** | **ValueError** |

**57 倍。** lift=4 被拒绝是对的：layer 0 的分母最大 0.2925，×4 = 1.17 > 1，
量程检查在 lift **之后**跑，所以越界会抛而不是悄悄算错。

---

## 1. 请注意的安全边界

**安全的 lift 取决于该层分母的最大值，每层不同。** layer 0 实测 max 0.3036，所以 3 安全（0.91）。
其它层我没量过。

`_check_inversion_range` **只在 ClearEngine 上能查**（它要读明文）。设备上没有这道保险。
所以：

> **先用 `--engine clear` 在同一个 lift 上跑一遍**，确认不抛，再拿去设备。

默认值是 **1**（完全保持原行为），我没有擅自改默认。

---

## 2. 和之前几件事的关系

| | 状态 |
|---|---|
| `--plaintext-cache` | 你已验证，13.4x。mask 也存盘了，热跑应再省约 38 s |
| `7a03a0d` 两 tower 编码 | **还没在你的构建里**（57.9 ms/次没变），重编译后冷跑 1112 s → 58 s |
| PADDING_FLOOR | 必要但不充分，保留；等你的逐槽 `<0` / `>1` 计数 |
| **`--inverse-lift`** | **新增，零 level 代价，clear 上就有 3–8 倍** |

这条**不解决发散**——发散还得靠 §PADDING_FLOOR 那条线和你的逐槽数据。
但它把"分母底部全是自举噪声"这个问题从 35.6 压到 0.63，是目前唯一不花 level 也不花时间的精度杠杆。

---

## 3. 建议这样跑

```
# 先确认 lift 对这层是安全的（会在越界时抛）
python3 -m thorfhe.bench fhe --engine clear --layers 1 --limit 1 --compact \
  --refresh-after-dense --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --inverse-lift 3

# 再上设备，带逐槽探针
THORFHE_DEBUG=1 python3 -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
  --per-stage --compact --refresh-after-dense \
  --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --extra-rotation-keys 6 --inverse-lift 3 --plaintext-cache /home/zhiyuan/ptcache
```

想要的是 `inv_iter01_b`–`inv_iter03_b` 三行的 `min` 和新的 `<0 n/N` / `>1 n/N` 计数。

217 passed, 67 skipped。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
