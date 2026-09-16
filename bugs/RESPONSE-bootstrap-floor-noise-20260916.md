# 确定性 + 固有底噪：不是放大器，bootstrap 自己产生 0.064

日期：2026-09-16。针对 `0b51b00`。

---

## 实验 3.1：同一密文两次 bootstrap（确定性检查）

```python
ct = engine.encrypt(zero)   # once
first  = decrypt(bootstrap(ct))
second = decrypt(bootstrap(ct))   # same ciphertext
```

**结果：**
```
|first - second| max = 0.000000e+00   ← 完全相同
|first| max = 6.291e-02, |second| max = 6.291e-02
```

**求值是确定性的。** 同一密文进去，两次 bootstrap 结果 bit-identical。
没有读取未初始化内存的问题。CKKS 求值无随机性，这与理论一致。

---

## 实验 3.2：噪声放大 vs 固有底噪

```python
fresh = engine.encrypt(zero)
noisy = engine.encrypt(zero)
for _ in range(1000):
    noisy = engine.rotate(noisy, 1)    # 累积 ~1.4e-9
```

**结果：**
```
input noise:       fresh 8.26e-10    after 1000 rotations 1.44e-09    ratio 1.74x
after bootstrap:   fresh 6.43e-02    rotated 6.61e-02                   ratio 1.03x
gain:              fresh 7.79e+07    rotated 4.60e+07
```

**输入噪声差 1.74x，输出几乎一样（ratio 1.03x）。**

按协作者 §3.2 的判据：
- 输出噪声跟着输入涨 → 放大器
- 两个都是 0.015 → 有自己的底噪，和输入无关

**结论：FLOOR。** bootstrap 有固有底噪 ~0.064，和输入噪声无关。
放大器假设不成立。增益 7.8e7 (2^26) 不是真的放大——输入只有 8e-10，
输出 0.064 是 bootstrap 自己产生的。

---

## 排除表更新

| 假设 | 状态 |
|---|---|
| alignToDiagonals 对角线 level 不一致 | ❌ 排除 |
| 误差是乘性/消息驱动 | ❌ 排除 |
| 周期 2048 结构 | ❌ 排除（统计假象） |
| key switch 本身有噪声 | ❌ 排除（单次 4e-10） |
| 确定性误差（常数表/明文编错） | ✅ 确认确定性（差 = 0） |
| bootstrap 放大输入噪声 | ❌ 排除（输出与输入噪声无关） |
| **bootstrap 有固有底噪 ~0.064** | **← 确认。底噪是确定性的，每次相同** |

---

## 重新理解"确定性 + 底噪"

bootstrap 输出 = 真值 + 固定误差 0.064。

这个 0.064 不是随机的——它是一个**确定性的、可复现的误差**。
对同一个密文，它每次都加同样的 0.064。

但它又不是"一个常数"——不同槽的误差不同（之前测到 p50=0.010, max=0.062），
分布是高斯形的。这说明它是某种**确定性的、槽位相关的误差模式**。

按协作者 §4 的线索，`Bootstrap.cu:250` 的 `constantEvalMult = pre / (k * N)`
把密文缩小 2^16 以上，Chebyshev 和 double-angle 在缩小后的尺度上做，
最后靠 `corFactor` 放大回去。**缩小期间的舍入误差是确定性的，
放大回去后表现为固定底噪。**

要确认：在 `multScalar(constantEvalMult)` 之后、`multIntScalar(corFactor)` 之前，
各解密一次看幅度——缩小了多少倍，就是误差被放大多少倍的上限。

---

## 测试

210 passed, 11 skipped（含修正后的 `test_bootstrapping_one_ciphertext_twice_gives_the_same_answer`
和 `test_bootstrap_noise_scales_with_the_input_noise`，CPU 下 skip bench params 的）。
