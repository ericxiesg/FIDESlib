# 按 q0/Δ 归一化之后,那 8 bit 的"模数效应"有一半是应该的

2026-09-24,针对 `82e9514`。

## 0. 结论先说

你们的扫描数据是对的,但"modulus size is the cause"这个读法**只对了一半**。

bootstrap 的误差**不是绝对量**——它把 `message_bound = q0/Δ` 复现到若干 bit。所以同一个实现换一组模数,原始误差本来就会变,**跨模数比原始 bit 数,测的是模数和实现两样东西**。

这不是我现在编的口径,是树里本来就有的模型:

```python
# thorfhe/clear.py:87
self.message_bound = 2.0 ** (first_mod_bits - scaling_bits)
# thorfhe/clear.py:152-153
"""A bootstrap reproduces ``message_bound`` to ``bootstrap_precision_bits``."""
return self.message_bound * 2.0 ** -self.bootstrap_precision_bits
```

把它反解出来(`effective_bootstrap_precision_bits`,本次提交):

| config | q0/Δ | 原始 bit | **归一化 p** |
|---|---|---|---|
| depth23 mod59/60 | 2 | 14.0 | **15.0** |
| depth23 mod50/55 | 32 | 6.0 | **11.0** |
| depth30 mod50/55 | 32 | 6.0 | **11.0** |
| depth37 mod50/55 | 32 | 6.0 | **11.0** |
| depth37 mod59/60 | 2 | 14.0 | **15.0** |
| depth37 mod52/55 | 8 | 10.6 | **13.6** |
| depth37 mod54/55 | 2 | 10.5 | **11.5** |
| **CPU mod50/55** | 32 | 33.1 | **38.1** |
| ClearEngine 默认假设 | — | — | 22 |

## 1. 这改变了结论

- 原始 bit 从 6.0 到 14.0,**8 bit 的摆动**,看起来模数影响巨大
- 归一化后从 11.0 到 15.0,**4 bit**

**那 8 bit 里有 4 bit 纯粹是 `q0/Δ` 从 32 掉到 2**,任何一个正确的 bootstrap 都会这样。这部分不是缺陷,是物理。

剩下的 4 bit 是真的,但比看起来小得多。而**真正压倒性的事实是:归一化精度在所有配置下都是 11–15 bit,而 CPU 在同参数下是 38.1**。

## 2. 所以要问的问题变了

不是"GPU 为什么在乎模数大小"(它基本不在乎),而是:

**为什么有效精度是 11–15 bit,而不是 38?**

这是一个**基本平坦的 23–27 bit 缺口**,跨深度、跨模数都在。平坦的缺口指向一个固定的结构性差异,不指向一个随参数缩放的数值退化——这和我在 `59f4d6c` 里说的"device 算术是精确整数、精度丢失这个机制不存在"是一致的。

## 3. 附带:mod52/55 那个点

归一化后 mod52/55 是 13.6,比 mod54/55 的 11.5 还高——**非单调**。原始数据里那个"plateau"在归一化之后变成了一个凸起。

7 个点、非单调、还有一个我看不出机制的凸起,我不建议在这上面拟合公式(你们写的 `2^-(scaling_bits - 44)` 我算过,50→6 对,52→8 但实测 10.6,不成立)。**先把第 4 节那一个数测了再说。**

## 4. 还是那个没测的数

`82e9514` 里:

> mod59/60: 33 - 14 = 19 bits (**assuming CPU also gets 33 bits with mod59/60**)

请把它测掉。按上面的框架,它问的是:**CPU 的归一化 p 是否也随模数变化?**

- CPU 在 mod59/60 归一化后仍是 ~38 → 缺口从 27 收到 23,**GPU 有轻微的模数依赖**
- CPU 在 mod59/60 归一化后更高 → 缺口恒定,**模数依赖完全是正常行为**,只需查那个常数缺口

一次 CPU 运行,不占显存。

## 5. 同时请跑 `tparams64_13_4_thormod`

`b8829e2` 加的那组:depth 23 + THOR 模数(50/55),既复现 6 bit 又装得下。它和 `tparams64_13_4_sparse`(59/60)只差模数,四个分段测试并排跑,**红的那一段就是落点**。

## 6. 报告口径建议

以后 bootstrap 精度请报**归一化后的 p**,或者两个都报。`thorfhe.clear.effective_bootstrap_precision_bits(rms, scaling_bits, first_mod_bits)` 已加好,`test_bootstrap_correction_factor.py` 已经改成两个都打。

不然换一次模数,数字就动,而动的原因一半不在被测对象身上。
