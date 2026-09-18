# 整模型两句话都对：encrypted accuracy 100%，标定是泛化的

日期：2026-09-18。承接 `7d227c4`。

---

## 0. 结论

十二层 + depth 41，ClearEngine：

```
encrypted run: 12 of 12 layers, 2 samples
  plaintext   accuracy 100.00%  F1 100.00%  (tp 1 fp 0 fn 0 tn 1, n=2)
  encrypted   accuracy 100.00%  F1 100.00%  (tp 1 fp 0 fn 0 tn 1, n=2)
  hidden after layer 11  MAE 3.179e-03  relRMSE 7.808e-03
  probabilities  L1 mean 3.046e-04  label agreement 100.00%
```

**两类都覆盖到了**（tp 1 / tn 1），不是蒙对一个类。

| | 1 句 | 2 句 |
|---|---|---|
| hidden relRMSE | 7.778e-03 | 7.808e-03 |

**跨句子稳定。**

另外跑了一次 5 句的：第 4 句时被宿主内存（我这台 8 GB 笔记本）杀掉，
但**前 3 句都跑完了十二层，零次量程告警**（`ValueError` / `ScaleMismatch` 计数都是 0）。

这一条是冲着"每层方差窗口是不是过拟到一句话"去的——**不是**。

---

## 1. 现在的完整状态

| | 状态 |
|---|---|
| 单层精度（ClearEngine） | relRMSE 2.784e-03 |
| 十二层精度（ClearEngine, depth 41） | relRMSE 7.8e-03，accuracy 100% |
| 十二层能不能装进 32 GB | **不能**，31.7 + keygen 3 GiB，差约 2.7 GiB |
| 设备单层精度 | **仍在查**（`he_inv` 负分母根因已定，floor 修了一半） |

---

## 2. 还等你两件事

1. **重编译带 `7a03a0d`**（两 tower 编码）。你最近一次的数字仍是 57.9 ms/次编码，
   说明 C++ 没重编。编完冷跑 1112 s → 58 s。
2. **`THORFHE_DEBUG=1 --per-stage` 的 `inv_iter01_b`–`inv_iter03_b` 三行**，
   要 `min` 和新的 `<0 n/N` / `>1 n/N` 计数。padding floor 修完之后设备上还剩的发散，
   我本机复现不出来，需要逐槽数据。

另外提醒：拉了 `42a43ed`（每层方差窗口）之后 **level 消耗变了，旋转密钥要重新生成**，
不能复用旧的。`--plaintext-cache` 不受影响。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
