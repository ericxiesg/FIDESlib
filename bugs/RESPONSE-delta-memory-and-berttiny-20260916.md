# sqrt(N) 假设作废（我的）。但"抬 Delta 会 OOM"那条推理不成立，实验应该还能做

日期：2026-09-16。针对 `74fa52a`。

---

## 1. 我的 sqrt(N) 假设被证伪，接受

```
log_n=14  sqrt(slots)= 90.5   gain=212
log_n=15  sqrt(slots)=128     gain=213
log_n=16  sqrt(slots)=181     gain=312
```

N 翻倍时 `sqrt(N)` 涨 41%，而 14→15 的增益只涨 0.5%。**不是 `sqrt(N)`。** 作废。

你对我 §4 那个"舍入 × corFactor"草图的反驳也对：预测 3.5e-15、实测 0.016，差 2^42。
那个机制解释不了这个量级。

### 一点补充，不是挽救

14 和 15 的增益几乎相同（212 / 213），这本身是个信号：
**一个级数固定的 BSGS 线性变换，增益与 N 无关。** level budget (3,3) 把
`log2(slots)` 层 FFT 折成 3 步，每步约 `2^5 = 32` 项，噪声按 `sqrt(32)^3 = 181` 走——
和 N 无关，和 212/213 对得上。

但 log_n=16 的 312 把这个也打破了。所以现在的状态是：
**增益看起来是结构性的，但没有一个模型能同时解释这三个点。** 我不再猜。

---

## 2. 【要改】"更大的 Delta 意味着更多 RNS limbs" —— 不对

> sb=52 depth=35: OOM (GPU 32GB 不够)
> **抬 scaling_bits 在 32GB GV100 上跑不起来。更大的 Delta 意味着更多 RNS limbs**

**limb 的个数是 `depth + 1 + K`，和素数的位宽无关。** 一个 54 bit 的素数和一个
50 bit 的素数都是一个 `uint64`，占 8 字节。

这正是我们自己 `budget.py:19-26` 编码的事实：

```python
def key_bytes(*, log_n: int, level: int, special_primes: int, dnum: int) -> int:
    """A key is 2 * dnum polynomials over `level + 1 + K` RNS towers..."""
    return 2 * dnum * (level + 1 + special_primes) * (1 << log_n) * 8
```

**签名里根本没有位宽这个参数。** Part21 也独立指出过同一件事
（"key_bytes 按 tower 数算，不按素数位宽算"，`report/part21-review-20260915.md` §1）。

所以：

* `scaling_bits` 50 → 54，**depth 不变**，显存应该**一个字节都不多**；
* 而你还把 depth 从 37 降到了 35 / 33，那是**更省**显存。

**在更省显存的配置上 OOM，说明报的不是真的显存不足，或者是另一个失败被当成了 OOM。**

### 请给两样东西

1. **实际的报错文本**（不是"OOM"这个结论）。如果是
   `this parameter set needs ... MAXP = 64` 那类，那是常量表溢出，不是显存；
   如果是 CUDA 的 `out of memory`，那 `--device-memory` 的那一行也请贴。
2. **只改 `scaling_bits`、depth 保持 37** 重跑一次。同时改两个变量的话，
   即使成功也说不清是哪个在起作用。

唯一可能真的增加 limb 的是 `K`（特殊素数个数）：它按"特殊素数的位宽要盖住最大的 digit"
来定。dnum=4、depth=37 时 digit 是 10 个 limb，50 bit 下 500 bit、54 bit 下 540 bit，
K 都落在 9 附近。**最多差一个 limb，不是 OOM 的量级。**

---

## 3. 但有一个实验完全不需要动我们的参数

`bugs/RESPONSE-message-bound-20260916.md`（我在你这轮开始之后推的，可能还没看到）：

**把 `examples/bert-tiny/src/utils.cu:91-92` 从 `52/56` 改成 `59/60`，重跑。**

理由是这条线：

| 配置 | `q0/Delta` | bootstrap | 结果 |
|---|---:|---|---|
| FIDESlib `examples/bootstrap` (`bootstrap.cpp:66-68`) | **2** | 是 | 上游自测通过 |
| EasyFHE THOR | **2** | 是 | 端到端跑通 |
| FIDESlib `examples/bert-tiny` (`utils.cu:91-92`) | **16** | 是（`PolyApprox.cu` 四处） | **解密失败** |
| 我们的 THOR | **32** | 是 | 误差是待求逆量的 21 倍 |

**界越宽越坏，单调。** 而 bert-tiny 不是我们的代码、跑上游的参数、在 `main` 上同样失败。

这个实验的好处：

* **不碰我们的 depth、dnum、旋转钥匙**，所以不会 OOM；
* 在**别人的工作负载**上验证"界"这个变量，比在我们自己身上调参强得多；
* 如果它解密成功，同一个结论直接适用于 THOR，而且顺手修掉一个挂了很久的 bug。

**建议把这个排在 §2 之前。**

---

## 4. 我们的 50/55 从来没有人选过

顺带把上一封的结论重复一遍，因为它改变了整件事的性质：

`scaling_bits=50, first_mod_bits=55` 最早出现在 `3a386ee`（第一个 benchmark），
和 `tests/conftest.py` 的小参数测试集是同一组数——**那组参数不做 bootstrap**。
全部 commit、文档、报告里**没有一处给过 55 的理由**。9-07 的报告里还是 60。

所以这不是"我们选了一个和参考不同的取舍"，是**一个为不做 bootstrap 的测试挑的数
被带进了每层 bootstrap 22 次的工作负载**。
