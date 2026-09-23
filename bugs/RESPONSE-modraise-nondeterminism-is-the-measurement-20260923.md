# ModRaise 的"非确定性"是测量方法造成的，不是 device bug

2026-09-23，针对 `556efcd` / `104f27d`。**请先不要按那两份报告的 next step 去查 `grow` / `generateSpecialLimbs` / `Accumulate`。**

---

## 0. 结论

`test_stage_stability.py` 和 `test_modraise_components.py` 都把 `engine.encrypt(b0)` **写在 run 循环里面**，所以每一轮从一个不同的密文开始。ModRaise 的输出依赖加密随机数，**每轮必然不同**——这是正确的 ModRaise 必须有的行为，不是缺陷。

"stage 1 之后 100% slot 不稳定"这个数字是对的，它测的东西不是非确定性。

---

## 1. 为什么 ModRaise 必然跟着加密随机数变

CKKS 加密是随机化的：`c1 = a` 是均匀随机的，`c0 = -a*s + Delta*m + e`。在 level 0 上

```
c0 + c1*s = Delta*m + e   (mod q0)
```

在整数上则是

```
c0 + c1*s = Delta*m + e + q0 * I
```

`I` 由这对多项式的实际系数决定，也就是由那个随机的 `a` 决定。

ModRaise 不计算任何新东西，它把同一对 `(c0, c1)` 重新解释到一个大得多的模 Q 上。于是它解密出来就是 `Delta*m + e + q0*I`，而 `q0*I` 这一项远大于 `Delta*m`。

**换一次加密 → 换一个 `a` → 换一个 `I` → 解密结果完全不同。**

把 `q0*I` 减掉正是 EvalMod 的职责，也是 bootstrap 里为什么需要 EvalMod。

报告里的数字本身就印证了这点：stage 1 的 slot0 是 `5.7e-05 / -0.0053 / 0.00136 / -0.00134`，输入 0.7058 根本看不见——因为它被 `q0*I` 淹没了。**如果 stage 1 在重新加密后仍然稳定，那才是要查的问题。**

---

## 2. "encrypt 是确定的"是阈值造成的

`test_encrypt_stability` 用 `spread > 1e-6` 判定。加密噪声在 scale 2^50 下约 1e-10，比阈值小四个数量级，所以 **encrypt→decrypt 往返**看起来是稳的。但底下的密文每次都不一样。报告把"往返稳定"读成了"密文确定"，ModRaise 随后把两者的区别暴露了出来。

---

## 3. stage 2 的 78.8% 是同一个东西传下去的

之前挂着的问题（stage-2 不稳定是真 race 还是读中间态的假象）现在有答案了：**是同一个测量缺陷**。stage 2 跑在 stage 1 的输出上，那个输出已经带着随机的 `q0*I`。

---

## 4. 正确的实验：密文只加密一次

`EvalBootstrap` **不会**改写它的入参。它在 `api/CryptoContext.cpp:1785` 做拷贝构造，而 `CiphertextImpl` 的拷贝构造在 `api/Ciphertext.cpp:34` 通过 `CopyDeviceCiphertext` **深拷贝了 device 密文**。所以

```python
ct = engine.encrypt(b0)                      # 一次
runs = [engine.bootstrap_stage(ct, stage) for _ in range(4)]   # 合法
```

是安全的，而且省掉了每轮的 keygen/encrypt 开销。

`python/tests/test_stage_stability_one_ciphertext.py` 已经写好，三个测试：

1. `test_each_stage_is_deterministic_from_one_ciphertext` —— 四个 stage 各跑 4 次，**按位相等**判定（不是容差）。每个 stage 都是定输入上的确定性算术（INTT / grow / broadcast / NTT / 旋转 / 明文乘），同一密文上必须给出同样的比特。用容差恰好会盖住你们要找的那种小 race。
2. `test_reencrypting_changes_modraise_and_that_is_correct` —— 对照组，复现旧数字，并且**反向断言**：如果重新加密之后 ModRaise 输出不怎么变，那才有问题。
3. `test_encrypt_decrypt_roundtrip_is_stable_but_the_ciphertexts_are_not` —— 说明第 2 节那个阈值效应。

运行：

```
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_stage_stability_one_ciphertext.py
```

**测试 1 的结果决定下一步。** 如果它全绿，device 是确定性的，`grow` / `generateSpecialLimbs` / `Accumulate` 都清白，非确定性这条线就此关闭，注意力回到精度本身（bootstrap 误差 σ = 0.0156，高斯，10.9 bit）。如果它红了，那是**真的** race，而且报告会直接给出是哪个 stage、哪个 slot、四轮各是什么值——比现在的线索精确得多。

---

## 5. 关于 weight cache：我们的结论一致

`88c290f` 说的对，而且补了一个我这边没有的事实——bootstrap 对角线走 `MakeCKKSPackedPlaintext`，根本不经过 light plaintext 路径，所以磁盘 cache 与 bootstrap 测试完全正交。

我这边独立查证的部分（`report/report-cts-stc-review-20260923.md` 第 3 节）：`7a03a0d` 的前提"FIXEDMANUAL 下系数与 level 无关"**早就有测试钉着**，不是只写在 commit message 里——`test_multiply_matches_dense_encoding` 比较的正是 dense 路径（fresh 密文 → level 0）和 light 路径（`towers-2`），容差 1e-9，cpu 和 cuda 都跑。所以 **THOR 的 9.5 GiB 权重 cache 也不需要重建**。

---

## 6. 另外：CtS/StC 测试之前没在跑我们的配置

`22ef07f` 已推。`TTALL64BOOT` 八组参数全是 FIXEDAUTO / FLEXIBLEAUTOEXT、depth 23、scale 59、uniform ternary；benchmark 是 FIXEDMANUAL、depth 37、scale 50、sparse ternary。其中两项直接落在被测代码里：`EvalCoeffsToSlots` 开头有一段 FIXEDMANUAL 专属的 rescale 从来没被触达，而 CoeffsToSlots 测试把 baby-step 切分钉死成 `{16,16}`，benchmark 传的是 `{0,0}` 让 OpenFHE 自己选——**测的是另一套对角线分解**。

现在加了 `tparams64_16_thor_fixmanual`，只挂在 bootstrap suite 上。麻烦重建后跑：

```
OpenFHEBootstrapTests/OpenFHEBootstrapTest.CoeffsToSlots
OpenFHEBootstrapTests/OpenFHEBootstrapTest.SlotsToCoeffs
```

把新那一组的 `Max error` 行连同原来八组一起贴回来。如果 depth 37 @ logN 16 在机器上放不下，把 `OpenFheInterfaceTests.cu:3853` 的 `TTALL64BOOTTHOR` 改回 `TTALL64BOOT` 就行，其余改动不依赖它。

同时 `AddBootstrapPlaintexts` 现在会校验从 OpenFHE 抄过来的对角线形状（层数、每层对角线数、层内 level 是否一致、层间是否恰好降一级、层序）。**尺寸不符会抛异常，level 关系只打印不抛**——"每步恰好消耗一级"是我对 OpenFHE level budget 的推断而非这个仓库声明过的东西，而我自己写的不变量检查已经误杀过两次正常 bootstrap。请把 `[FIDESlib] bootstrap precomputation:` 开头的行贴回来。
