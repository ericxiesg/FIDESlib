# 剩下的 level 在哪里：he_inv 占 14/36，但那个界是推出来的，我不动它

日期：2026-09-18。承接 `81108ad`。

---

## 0. 结论

整模型要 depth 41，装不下（差约 2.7 GiB）。所以我去找还能省的 level。
`_check_inversion_range` 现在在 `THORFHE_DEBUG=1` 下**报告实际观测到的量程**，
不只是越界才抛。十二层跑一遍（depth 41）得到 26 次 `he_inv` 调用的实测：

```
[range] he_inv observed [0.0264306, 0.303622] ratio 0.0870511 against epsilon 0.015625
[range] he_inv observed [0.0232071, 0.306772] ratio 0.0756494 against epsilon 0.00390123
...
```

**`he_inv` 十二层总共 166 次迭代 = 每层约 14 个 level**（一层一共 36 个）。
如果把 epsilon 收到实测比值（margin 2），是 **130 次，每层省 3.0 个 level**。

那大概等于 depth 41 → 38，也就是 **1.4 GiB**；再加上退回 binary 旋转密钥的 1.8 GiB
（你实测只值 7 秒），整模型就装得下了。

---

## 1. 但我没有动它，原因

那个 epsilon 不是标定值，是**推导出来的界**（`softmax.py:124`）：

```python
epsilon = precision / 128 / 2
```

`precision` 是上一次 `he_inv` 达到的下界，`128` 是 token 数 `g.dim`——
这是"n 个数之和为 S，其平方和至少 S²/n"那一类的界，**不是随手写的常数**。

而越界的后果恰恰是**不报错**：`_check_inversion_range` 的 docstring 写着，
低于下界 Goldschmidt **饱和**——epsilon/2 时低 3.6%，epsilon/10 时低 56%，
再往下就钉在 11500 不动了，**返回一个有限的、看起来合理的错数**。设备上没有这道检查。

所以把它收紧需要**重做那个推导**，或者拿很多句话在 FHE 上重新标定。
我手上只有**一句话、一个 depth** 的观测。用这个去换 3 个 level，是拿正确性换空间，
证据不够。**这一条我标出来但不做。**

（LayerNorm 那次不一样：方差是在**明文模型**上量的，200 句话，几秒钟，
和 FHE 的 level 预算无关，所以那个标定是便宜且可复现的。）

---

## 2. 所以现在的账

| 省 level 的手段 | 每层 | 状态 |
|---|---|---|
| 每层方差窗口 | **4** | ✅ 已做（`42a43ed`） |
| `he_inv` 的 epsilon 收紧 | 3.0 | ⚠️ 界是推导的，需要重做推导或多句标定 |
| `--inverse-lift` 当 level 用 | 0.2 | ❌ 量过了，不值（实测分母上界多在 0.4–0.99，安全 lift 是 1） |

| 省显存的手段 | GiB | 状态 |
|---|---|---|
| 退回 binary 旋转密钥（21 把 → 15 把） | **1.8** | ✅ 随时可做，你实测只值 7 秒 |
| depth 41 → 38（靠上面省 3 个 level） | 1.4 | 依赖上一格 |

**binary 那一条是现在就能拿的 1.8 GiB，而且几乎免费。**

---

## 3. 顺带：`--inverse-lift` 作为 level 手段被否掉了

我原本以为它既提精度又省 level（epsilon 变大 → 迭代变少）。**量了，不是**：
十二层实测分母上界大多在 0.4–0.99，安全 lift 只有 1，全程只省 2 次迭代。

它**作为精度手段仍然成立**（layer 0 实测 3–8 倍），只是别指望它省 level。

---

## 4. 新增的调试输出

`THORFHE_DEBUG=1` 下，`he_inv` 和 `he_invsqrt` 每次调用都会打一行实测量程。
标定这两个迭代的时候用得上，设备上也能用（它走的是 ClearEngine 的可读路径，
所以设备上不会打——那边靠探针的 p50）。

221 passed, 67 skipped。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
