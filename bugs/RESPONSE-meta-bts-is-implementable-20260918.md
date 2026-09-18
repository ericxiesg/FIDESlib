# Meta-BTS 在 FIDESlib 里能实现，不用动 CUDA —— 已实现并量过

日期：2026-09-18。

---

## 0. 结论

| 问题 | 答案 |
|---|---|
| `EvalBootstrap(numIterations=2)` 能用吗 | **不能**，GPU 路径把参数丢了（`CryptoContext.cpp:1763-1778` 只在 CPU fallback 转发） |
| 需要写 CUDA 吗 | **不需要**。所需原语全部已导出 |
| 实现了吗 | 是，`Stages.bootstrap_twice`（`stages.py`），~12 行 |
| 代价 | **2 次 bootstrap + 1 个 level** |
| 收益 | **正好 `k` bit**，`k = meta_bts_bits` |

---

## 1. 实测（ClearEngine + noise model，误差设为设备实测精度）

```
   bits   k   single BTS     Meta-BTS     gain   levels
     15   -    2.530e-04            -        -        0
     15   6    2.530e-04    4.017e-06      63x        1
     15  10    2.530e-04    2.511e-07    1008x        1
     15  13    2.530e-04    2.442e-04       1x        1   <- 越界
     20   6    7.905e-06    1.255e-07      63x        1
     20  10    7.905e-06    7.846e-09    1008x        1
     20  13    7.905e-06    9.806e-10    8062x        1
```

增益正好是 `2^k`。**设备实测 15.2 bit，取 k=10 → 等效 25 bit。**

---

## 2. k 的上限

被抬起来的残差必须留在正弦能恢复的范围内：

```
2^k * |e| < q0/(2*Delta)
```

15 bit 时 `|e| = 2.53e-04`，界是 1.0 → `k <= 11`。上表 k=13 那行是 `2.07 > 1.0`，
残差被替换成余数，**增益归零**（不是变小，是没有）。取 k 时要按实测误差算，不能拍。

---

## 3. 实现

```python
def bootstrap_twice(self, x, keep_levels=None):
    first = self.bootstrap(x, keep_levels)                       # x + e
    drop = self.engine.level(first) - self.engine.level(x)
    at_input = self.engine.level_down(first, by=drop) if drop > 0 else first
    residual = self.subtract(x, at_input)                        # -e
    lifted = self.multiply(residual, 2 ** self.meta_bts_bits)    # 整数乘：0 level
    second = self.bootstrap(lifted, keep_levels)                 # -2^k e + e'
    corrected = self.rescale(self.multiply(second, 2.0 ** -self.meta_bts_bits))
    left, right = self.align(first, corrected)
    return self.add(left, right)                                 # x + 2^-k e'
```

用到的原语和它们在 FIDESlib 里的落点：

| 原语 | FIDESlib |
|---|---|
| `bootstrap` | `EvalBootstrap` ✅ |
| `level_down(ct, by)` | `EvalLevelReduce` ✅ |
| `level(ct)` | `GetRemainingLevels` ✅ |
| `subtract(ct, ct)` | `EvalSub` ✅ |
| `multiply(ct, int)` | `EvalMultByInteger` ✅ **0 level** |
| `multiply(ct, float)` + `rescale` | `EvalMultScalar` + `Rescale` ✅ |
| `add` | `EvalAdd` ✅ |

**一个都不缺。** 把 `level_down` 用来回到输入的模数是关键那步——
`b0` 在链顶、`x` 在链底，要相减必须先把 `b0` 降回去。

---

## 4. 用不用它

**现在的证据说不需要**：设备上 bootstrap 是 15.2 bit，而发散点在 `norm_1`，
noise model 在同样的误差下不炸。Meta-BTS 解决的是精度，而当前的病不是精度。

但它值得留着，因为：

1. 如果 `norm_1` 查下来确实需要更高精度，这条路现在是通的，改一个常量就能用；
2. 每层 22 次 bootstrap 全用 = +22 次 bootstrap +22 level，**太贵**；
   按站点用才合理——比如只给 `he_invsqrt` 的输入用，那是 1 次；
3. 它给了一个不用 GPU 就能回答"精度够不够"的办法：把 `meta_bts_bits` 调上去，
   如果结果不变，就证明精度不是瓶颈。

---

## 5. 测试

`test_meta_bts_buys_its_bits_back`：k=6 和 k=10 各自给出 `2^k`（±15%），各花 1 个 level；
k=13 在 15 bit 下越界，断言它**不再有增益**——这条是为了钉住上限不是渐变而是突变。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
