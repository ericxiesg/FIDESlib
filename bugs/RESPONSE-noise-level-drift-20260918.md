# noise_level 核查：rescale 的守卫确实被注释掉了，但复现方式和流水线不一样

日期：2026-09-18。针对 `4f169e6`。

---

## 0. 结论

| 你的说法 | 核查 |
|---|---|
| `rescale` 让 noise_level 变负 | **对**，而且守卫被注释掉了（`Ciphertext.cpp:499`） |
| uint32 下溢 | **不是**。`NoiseLevel` 是 `int`（`Ciphertext.cuh:81`）。4294967285 是 binding 把负 int 转成无符号的显示 |
| 连续 rescale 复现 | **不是流水线做的事**。这条复现不能用来定位 norm_1 |
| 选项 B（encode 时匹配密文 noise_level） | **不要做**，会让 subPt 接受真正不匹配的 scale |
| 选项 D（rescale 不该变负） | **对，就是它** |

---

## 1. 守卫被注释掉了

```cpp
// src/CKKS/Ciphertext.cpp:496
void Ciphertext::rescale() {
    // assert(this->NoiseLevel == 2);        <- 注释掉了
    ...
    NoiseLevel -= 1;                          // 无条件
    assert(NoiseFactor == (NoiseLevel == 1 ? ScalingFactorReal : ScalingFactorRealBig));
}
```

对一个已经 canonical（NoiseLevel==1）的密文调 rescale，会静默变成 0、−1、−2……
release build 里 518 行那个 assert 也被编掉，所以一路无声。

**这是真 bug，无论 norm_1 是不是它引起的都该修。**

---

## 2. 但你的复现不是流水线做的事

你 §2 是 encrypt 之后**连续 rescale**，中间没有乘法。流水线不这么做：
每次 rescale 前面都有一次把 degree 抬到 2 的乘法。

`ClearEngine.rescale`（`clear.py:337`）**本来就拒绝**对 canonical 密文 rescale：

```python
if self.strict and ct.scale_exp < 2:
    raise ScaleMismatch("rescale: ciphertext is already canonical (scale D^1); ...")
```

而 ClearEngine 跑完整层（含 stage 11 和 `he_invsqrt`）**不抛**。
所以 **Python 侧从来没有对 canonical 密文 rescale 过**。

---

## 3. 真实跑里 22 次 bootstrap 都验过 noise_level == 1

`pyfideslib.bootstrap`：

- 入口：`if entering != 1: raise ValueError(...)`
- 出口：rescale 循环之后 `if remaining != 1: raise RuntimeError(...)`

一层 22 次，**都没抛**。所以真实跑里每次 bootstrap 前后 NoiseLevel 都正好是 1。

`norm_1` 之前最近的 bootstrap 是 `refreshed_dense`（level 20，4 次）。
从那里到 `he_invsqrt` 的 subtract 之间是 `he_layernorm`，逐个核过：

| 操作 | NoiseLevel |
|---|---|
| `rescale(multiply(ct, values))` | 1→2→1 ✅ 配对 |
| `rescale(multiply(fold(total), statistic))` | 1→2→1 ✅ |
| `rescale(relinearize(square(total)))` | 1→2→1 ✅ |
| `multiply(ct, n)`（n=768，整数，**不 rescale**） | `multIntScalar` 走 `requireDegreeOne`，**不碰 NoiseLevel** ✅ |
| `subtract(ct, ct)` / `add` | 不变 ✅ |

C++ 侧也核过：`multPt` 用 `multMetadata` 置 `a.NoiseLevel + b.NoiseLevel = 2`，
融合 rescale 再减回 1 ✅；`multIntScalar` 不动 NoiseLevel ✅。

**没有一处是「减而不加」的。**

---

## 4. 所以要确认一件事

你 §1 那条错误信息：

```
subPt with mismatched scales - the ciphertext is at noise level -11 and the plaintext at 1
```

**是真实那次 layer 0 跑出来的，还是 §2 合成测试跑出来的？**

- 如果是**合成测试**：那 §1 的根因不成立，norm_1 的原因还没找到，
  因为流水线不会连续 rescale。
- 如果是**真实跑**：那就有一处 GPU 侧元数据和 ClearEngine 不一致，
  而且它在 `refreshed_dense` 和 `he_invsqrt` 之间——上面那张表里的某一行，
  GPU 的行为和我读到的代码不一样。

这两种情况下一步完全不同，所以先确认这个。

---

## 5. 建议的修法

### 立刻做：把守卫恢复成运行时错误

```cpp
void Ciphertext::rescale() {
    if (cc.rescaleTechnique == FIXEDMANUAL && NoiseLevel != 2) {
        OPENFHE_THROW("rescale: ciphertext is at noise level " + std::to_string(NoiseLevel) +
                      ", not 2 - rescaling it would put the scale below Delta");
    }
    ...
}
```

不是 `assert`（release build 会编掉），是真的抛。

**代价为零**：如果流水线本来就不这么做（ClearEngine 证明了 Python 侧不这么做），
这个守卫永远不触发；一旦触发，它直接指出是哪个操作，而不是十几个 level 之后
在 `subPt` 上表现成一条看不懂的消息。

**而且它能直接回答第 4 节的问题**：加上之后跑一次 layer 0，
抛了就说明真实跑里确实有非法 rescale，并且当场告诉你在哪。

### 不要做：选项 B

`encode(x, level=..., noise_level=self.noise_level(y))` 会让 `subPt` 的检查通过，
但那个检查是对的——密文的 scale 真的是 `Delta^-11`，明文是 `Delta^1`，
相减在数学上就是错的。让它通过只会把错误从「报错」变成「静默的错数」。

---

## 6. 我这边的状态

- `he_invsqrt` 的范围检查已加（`_check_invsqrt_range`），能区分「方差跑出窗口」和「算术错」。
- `subtract(ndarray, ct)` 的测试已加（`test_subtract_a_ciphertext_from_a_plaintext_vector`），
  本机 skip，等你那边跑。
- Meta-BTS 已实现（`Stages.bootstrap_twice`），2 次 BTS + 1 level 换 k bit，未开启。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
