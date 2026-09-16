# `q0/Delta = 32` 从来没有人选过它。而且它可能同时解释 bert-tiny

日期：2026-09-16。

---

## 1. 我们的 50/55 是继承来的，不是选出来的

查了全部历史，**没有任何一个 commit、文档或报告给过 `first_mod_bits = 55` 的理由。**

* `scaling_bits=50, first_mod_bits=55` 最早出现在 `3a386ee`（2026-09-04，第一个 benchmark）；
* 它和 `python/tests/conftest.py` 的小参数测试集是同一组数
  （`log_n=13, depth=12, scaling_bits=50, first_mod_bits=55, dnum=3`）——**那组参数不做 bootstrap**；
* 9-07 那份保真度报告里 `first_mod_bits` 还是 **60**，9-08 那份变成 **55**，中间没有说明。

**也就是说：一组为"不做 bootstrap 的功能测试"挑的参数，被带进了一个每层做 22 次 bootstrap 的工作负载。**

---

## 2. FIDESlib 自己的 bootstrap 例子用的就是 2

`examples/bootstrap/src/bootstrap.cpp:66-68`：

```cpp
ScalingTechnique rescaleTech = FLEXIBLEAUTO;
uint32_t dcrtBits            = 59;
uint32_t firstMod            = 60;
```

**`q0/Delta = 2^(60-59) = 2`** —— 和 EasyFHE THOR 的 `RESCALE_PRIME_BITS=59, FIRST_PRIME_BITS=60`
一模一样。这是 CKKS bootstrap 的常规选择，不是巧合。

而用 `SetScalingModSize(50)` 的是 `examples/advanced`，**那些例子不做 bootstrap**。

---

## 3. 排开看，是一条单调的线

| 配置 | `q0/Delta` | 做 bootstrap | 结果 |
|---|---:|---|---|
| FIDESlib `examples/bootstrap` | **2** | 是 | 上游自测通过 |
| EasyFHE THOR (`config.py:72-73`) | **2** | 是 | 端到端跑通，rel-L2 ≤ 5e-2 |
| FIDESlib `examples/bert-tiny` | **16** | 是（`PolyApprox.cu` 四处） | **解密失败**："approximation error is too high" |
| 我们的 THOR | **32** | 是（22 次/层） | bootstrap 绝对误差 1.04e-2，是待求逆量的 **21 倍** |

**界越宽，越坏，而且是单调的。**

这条线的分量在于：**bert-tiny 不是我们的代码。** 它是上游的例子，用上游的参数，
在 `main` 上同样失败（`BUG-bert-tiny-decryption-fails-20260914.md` 的测试矩阵第 3 行）。
我此前把它归档为"不是我们的问题"——**如果界就是原因，那它和我们的 softmax 是同一个 bug。**

---

## 4. 一个实验能同时验证两件事

**把 bert-tiny 的 `utils.cu:91-92` 从 `52/56` 改成 `59/60`，重跑。**

```cpp
constexpr uint32_t scale_mod_size = 59;   // was 52
constexpr uint32_t first_mod      = 60;   // was 56
```

* **解密成功** → 界就是根因。那么同一个修法直接适用于 THOR，
  而且是在**另一个工作负载**上独立验证的，比在我们自己身上调参强得多；
* **仍然失败** → 界不是（唯一的）原因，bert-tiny 另有问题，我们回到 §5 那个实验。

这比我上一封要的"单独抬 `scaling_bits`"更值，因为它一次回答两个 bug。
**建议先跑这个。**

（bert-tiny 用 FLEXIBLEAUTO、depth 23-25，和我们的配置很不一样，所以它不是我们的
代理——但如果界能修好它，那界对 bootstrap 精度的作用就坐实了。）

---

## 5. 如果 §4 不成立，再做原来那个

单独抬 `scaling_bits`（50 → 55，`first_mod_bits` 提到 60 以上），看绝对误差
是否按 `1/Delta` 下降。先前那次只动 `q0` 没动 `Delta`，误差没变；这是另一半。

---

## 6. 但有一个数，无论如何都要面对

即使 §4 成立、我们迁到 `q0/Delta = 2`，还有一件事拦着：

**我们 stage 15（`layernorm.py:172`）送进 bootstrap 的幅度是 17.57**
（`RESPONSE-bootstrap-precision-20260915.md` §4.1 实测，一层 22 次里有 4 次）。
新界只有 2，**超出 8.8 倍**。

而 THOR 在同一处必须保持在 2 以下。所以两者之间有一个 ~9 倍的差距要解释：

* 要么 THOR 的 stage-15 残差本来就是 O(1)，而**我们上游某处的缩放不对**；
* 要么 THOR 在那里另有补偿，而我们照搬了它的"有意加倍"却没照搬补偿。

`stage_15_prepare_layernorm` 的 docstring 说加倍是有意的、下游 `variance_window`
接受四倍方差来补偿。**那解释了方差，没有解释幅度。**

这一条不依赖 §4 的结果，可以并行查：把 THOR 在 stage 15 前后的实际幅度量出来对比。
我这边可以用 `ClearEngine` 跑真实 checkpoint 量我们的那一侧。

---

## 7. 顺带修正我自己先前的一个判断

我在 `report/part21-review-20260915.md` §4 反对 Part21 的"Delta 50→59, q0 55→60"，
理由是它把界压到 2 而 stage 15 超界。

**那个反对把因果弄反了**：不是"界太小所以放不下 17.57"，而是
**"界一直太宽，所以 17.57 一直没被发现是个问题"**。上游的 bootstrap 例子和 THOR
都用 2，我们用 32 没有任何理由——宽界不是我们的余量，是我们的 bug 在藏身的地方。
