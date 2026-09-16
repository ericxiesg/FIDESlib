# 两个实验完成：噪声确认，key switch 健康

日期：2026-09-15。针对 `d80f7b9`。

---

## 实验 1：同输入跑两次 bootstrap，看相关系数

```python
zero = np.zeros(32768, dtype=complex)
first  = decrypt(bootstrap(encrypt(zero)))
second = decrypt(bootstrap(encrypt(zero)))
correlation = corrcoef(first, second)
```

**结果：**
```
correlation = -0.002176
|first - second| max = 8.70e-02
|first| max = 7.14e-02,  |second| max = 6.26e-02
```

**相关系数 ≈ 0 → 是随机噪声，不是确定性误差。**

按协作者 §3 的判据：
- 相关系数 ≈ 0 → 每次运行不同 → 随机噪声 → 我的读法成立
- 相关系数 ≈ 1 → 确定性误差 → 可以直接减出来看

**结论：噪声。** 不是某个常数表/明文编错，是某处算术在每一步注入随机误差。

---

## 实验 2：单次旋转的精度

```python
x = rand(77, scale=0.4)
rotated = decrypt(rotate(encrypt(x), 1))
error = max(|rotated - roll(x, -1)|)
```

**结果：**
```
single rotation error = 4.23e-10
```

**key switch 健康。** 单次旋转误差 4e-10，远小于 bootstrap 的 0.015。

按协作者 §4 的判据：
- ~1e-13 → key switch 健康，噪声出在 bootstrap 自己的算术里
- ~1e-3 → key switch 本身就坏了

实测 4e-10，在 1e-13 和 1e-6 之间。**key switch 健康，不是噪声来源。**
一层 4000-8000 次旋转累积后也远小于 0.015（4000 × 4e-10 ≈ 2e-6）。

---

## 排除总结

| 假设 | 状态 |
|---|---|
| alignToDiagonals 对角线 level 不一致 | ❌ 排除（无 warning） |
| 误差是乘性/消息驱动 | ❌ 排除（零密文和全 1.0 分布相同） |
| 周期 2048 结构 | ❌ 排除（是分组样本数的统计假象） |
| 确定性误差（常数表/明文编错） | ❌ 排除（相关系数 ≈ 0） |
| key switch 本身有噪声 | ❌ 排除（单次旋转 4e-10） |

**剩下的方向：bootstrap 内部的算术在某处注入随机噪声。** 可能的位置：
1. Chebyshev 多项式系数表编码精度
2. double-angle 迭代的中间值精度
3. CtS/StC 对角线明文的编码精度（不是 level 不一致，而是编码本身的舍入）
4. ModRaise 后的 correction factor 计算

---

## 测试

210 passed, 9 skipped（含新加的 `test_bootstrap_noise_differs_between_two_runs` 和
`test_a_single_rotation_is_accurate`，CPU 模式下 skip bench params 的那个）。
