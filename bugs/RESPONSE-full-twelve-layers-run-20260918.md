# 整个 BERT-base 十二层跑通了（ClearEngine），depth 41；但 depth 41 装不进 32 GB

日期：2026-09-18。承接 `42a43ed`。

---

## 0. 结论

加上层边界 refresh（`2e78bcd`）和每层方差窗口（`42a43ed`）之后，**十二层第一次完整跑通**：

```
LAYER  0: in level 41 -> out level  9   |x| max 19.44
LAYER  1: in level  9 -> out level  7
...
LAYER  9: in level  7 -> out level  1   |x| max 39.07     <- 最紧的一层
LAYER 10: in level  1 -> out level  3   |x| max 49.91
LAYER 11: in level  3 -> out level  9   |x| max 18.03

hidden after layer 11  MAE 3.442e-03  relRMSE 7.778e-03
probabilities  L1 7.131e-06   label agreement 100.00%
```

单层是 2.784e-03，十二层是 7.778e-03——误差按层缓慢累积，**标签全对**。

---

## 1. 每个 depth 能跑几层

| depth | 跑通层数 | 最紧时的 level |
|---|---|---|
| **37**（现在用的） | **8**（layer 8 挂） | 1 |
| **41** | **12** ✅ | 1（layer 9） |
| 45 | 12 ✅ | 5（layer 9） |

**depth 41 是整模型的下限**（layer 9 掉到 1，再低就没有了）。depth 37 只能跑 8 层。

---

## 2. 但是 depth 41 装不下

`bench budget --keys 21`：

| depth | rotation keys | bootstrap keys | bootstrap plaintexts | 合计 | 32 GB 卡余量 |
|---|---|---|---|---|---|
| 37 | 4.0 | 9.2 | 7.6 | **29.8 GiB** | 2.2 GiB |
| **41** | 4.3 | 9.9 | 8.4 | **31.7 GiB** | **0.3 GiB** |
| 45 | 4.7 | 10.7 | 9.2 | 33.5 GiB | **−1.5 GiB 装不下** |

而 `AddRotationKeys` 的分解 scratch 要 **3 GiB**。

**所以：整模型需要 depth 41，depth 41 只剩 0.3 GiB，keygen 要 3 GiB。差约 2.7 GiB。**

这就是整个项目一直在打的那一仗，现在有了确切数字：
**不是"精度修好就能跑整模型"，而是还差 2.7 GiB 或者等价的 level。**

---

## 3. 往下走的方向（按我现在的判断排序）

1. **继续压 level。** 每层方差窗口刚刚省了 4 个（layer 0 出口从 level 1 到 5）。
   同样的手法还没用在 softmax 上：`inv_epsilon` 是每层的，但不知道紧不紧——
   如果和方差窗口一样松，那里还有 level。**每省 4 个 level 大约等于 depth −4 ≈ −1.9 GiB。**
2. **`--extra-rotation-keys` 调小。** 21 把是 4.3 GiB，binary 15 把是 2.5 GiB，省 1.8 GiB，
   代价是旋转 3643 → 8258——而按你的实测那只值 **7 秒**。**这条几乎是白送的。**
3. bootstrap level budget (3,3) → 其它组合，换 bootstrap plaintexts 的 8.4 GiB。

2 和 1 加起来就够了。

---

## 4. 对你当前工作的影响

**没有**——你在跑单层 depth 37，这条路径没变。

但它把"还差多远"说清楚了：单层精度修好之后，整模型还需要 depth 41 和约 2.7 GiB 的腾挪。
好消息是 §3 的第 2 条（少用旋转密钥）按你自己的实测只值 7 秒，所以基本是免费的。

---

## 5. 复现

```
python3 -m thorfhe.bench fhe --engine clear --compact --layers 12 --limit 1 --depth 41 \
  --refresh-after-dense --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16
```

本机 8 GB 内存跑得动，约 12 分钟。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
