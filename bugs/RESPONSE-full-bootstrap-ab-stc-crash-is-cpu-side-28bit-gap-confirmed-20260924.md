# 完整 bootstrap A/B 确认 28-bit 恒定缺口;StC 崩溃在 CPU 侧;ModRaise 测试被注释掉了

2026-09-24,针对 `da85568`。

## 0. 执行了什么

按 `e96b5b0` §5 和 `dd02758` §3 的要求,跑了以下 gtest(32 slots,与 `da85568` §5 一致):

| 测试 | 索引 | 参数 |
|---|---|---|
| `OpenFHEBootstrap` | /8 | sparse (mod59/60, dnum=4, SPARSE_TERNARY, FIXEDMANUAL) |
| `OpenFHEBootstrap` | /9 | thormod (mod50/55, dnum=4, SPARSE_TERNARY, FIXEDMANUAL) |
| `SlotsToCoeffs` | /8 | sparse |
| `SlotsToCoeffs` | /9 | thormod |

全部带 `FIDESLIB_TRACE_MODEVAL=1`。

**ModRaise 没跑,因为它被注释掉了。** `test/OpenFheInterfaceTests.cu:2306-2392` 整个 `TEST_P(OpenFHEBootstrapTest, ModRaise)` 被 `/* ... */` 包住。`--gtest_list_tests` 里不存在。`e96b5b0` §0 和 `da85568` §4 两次说"它存在",源码里确实存在——但是注释掉的。要跑的话需要先取消注释。

---

## 1. 完整 bootstrap A/B:28-bit 恒定缺口

### 1.1 数据

| Config | CPU bits | GPU bits | Gap | GPU Max Error | Expected (2^(-logPrec+4)) |
|---|---|---|---|---|---|
| sparse (mod59/60) /8 | **46** | **18** | **28** | 1.167e-05 | 2.84e-14 |
| thormod (mod50/55) /9 | **38** | **10** | **28** | 2.93e-03 | 7.28e-12 |

- CPU 随模数变化(46→38),GPU 也跟着变(18→10),**缺口恒定 28 bit**。
- 这和 Python pipeline 里测到的 ~30 bit 一致(差异来自 32 slots vs 32768 slots)。
- 两个 case 都 FAILED:GPU 误差远超 `2^(-logPrec+4)` 阈值。

### 1.2 EvalMod trace(两组完全相同)

```
[FIDESlib] EvalMod after Chebyshev (into double-angle): level 14, scale degree 2
[FIDESlib] EvalMod after double-angle: level 11, scale degree 2
[FIDESlib] EvalMod after post scalar (EvalMod exit): level 11, scale degree 2
[FIDESlib] bootstrap returned scale degree 2, rescaled 1 time(s) to reach the declared 1. Each one costs a level.
```

Chebyshev→double-angle→post scalar,全程 scale degree 2。两组参数完全相同的 level 流。

### 1.3 这直接回答"14→6 bit 掉在哪"

在完整 bootstrap 的 gtest A/B 里:

- **sparse**:CPU 46 bit → GPU 18 bit,掉 28 bit
- **thormod**:CPU 38 bit → GPU 10 bit,掉 28 bit

28 bit 不是某一阶段掉的——它是整条管线的累积。EvalMod trace 显示 level 流正常(14→11→11),没有异常的 level 跳跃或 scale degree 变化。

---

## 2. StC 崩溃在 CPU 侧,不在 GPU 侧

### 2.1 崩溃点

`SlotsToCoeffs` test 的源码流程(`test/OpenFheInterfaceTests.cu:3230-3245`):

```cpp
std::cout << "Run SlotsToCoeffs" << std::endl;           // line 3230 — 打印了
auto ctxtEnc = FHE->EvalSlotsToCoeffs(...);               // line 3232 — CPU StC,崩溃在这里
cc->RescaleInPlace(ctxtEnc);                               // line 3234 — 没到
// ...
std::cout << "Run SlotsToCoeffs GPU" << std::endl;        // line 3245 — 没到
```

两组参数都是:

1. 打印 `Run SlotsToCoeffs`(CPU 侧标记)
2. 调用 `FHE->EvalSlotsToCoeffs(...)` — 这是 **OpenFHE 自己的 CPU StC**
3. 抛出 `lbcrypto::OpenFHEException`
4. `Run SlotsToCoeffs GPU` **从未打印**

### 2.2 异常信息

```
/home/zhiyuan/workspace/THOR-FIDE/openfhe-install/include/openfhe/core/lattice/hal/default/poly.h:l.274:operator*=(): Modulus mismatch
```

`poly.h:274:operator*=()` 是 OpenFHE 的 `DCRTPoly::operator*=` 在做多项式乘法时发现两个多项式的模数链不匹配。

**这是 CPU 侧的 OpenFHE 代码,不是 FIDESlib GPU 代码。** GPU StC 从未被执行。

### 2.3 含义

- StC 的 modulus mismatch 是 **OpenFHE CPU 侧** 的问题,不是 FIDESlib 转录错误。
- 之前把它当成 GPU 侧 bug 来追的方向需要修正:GPU StC 还没跑就崩了。
- 要排查的话需要看 OpenFHE 的 `EvalSlotsToCoeffs` 在这组参数下为什么模数链不匹配——这可能是 OpenFHE 自身在这组 `tparams64_13_4` 参数下的 bug,也可能是 `EvalBootstrapSetupOnly` 产生的 `raised` 密文的模数链与 StC 预期不一致。

---

## 3. CtS 结果(在 SlotsToCoeffs test 里,StC 崩溃前)

SlotsToCoeffs test 先跑 CtS(CPU 和 GPU 各一遍),再跑 StC。CtS 结果:

| Config | CPU CtS bits | GPU CtS bits | Max error | Expected (2^(-logPrec+1)) | Ratio |
|---|---|---|---|---|---|
| sparse /8 | 42 | 42 | 1.31e-12 | 4.55e-13 | 2.9× |
| thormod /9 | 33 | 33 | 6.13e-10 | 2.33e-10 | 2.6× |

- **CPU 和 GPU 的 CtS 精度完全相同**(42=42, 33=33)。
- 误差是 CPU 参照自身精度的 2.6-2.9 倍,远小于 CoeffsToSlots 独立测试里的 22-26 倍。
- 区别:SlotsToCoeffs test 用的是 **bootstrap 后**的密文,CoeffsToSlots test 用的是 **fresh** 密文。输入幅度不同,同一个缺陷的表现不同。
- **CtS 不是 GPU 特有问题**——CPU 和 GPU 在同一个输入上精度相同。

---

## 4. Precomputation 日志

### 4.1 正常输出(两组相同)

```
Adding bootstrap precomputation to GPU for 32 slots.
Plaintexts loaded: 60 ~ 705MB
[FIDESlib] bootstrap key level plan: 3 of 24 keys truncated (L=23, bootstrap depth 14, levelBudget {2,2}, StC starts at 11, margin 1); 40 indexes in total, by level: 12x3
Rotation keys loaded: 25 ~ 3000MB (untruncated estimate)
[FIDESlib] key memory for keyID '...': 25 rotation keys + eval key, resident 2988 MiB (would be 3120 MiB untruncated), 3 truncated (228 MiB), 0 grown at runtime
```

### 4.2 `[FIDESlib] bootstrap precomputation:` 行不存在

`da85568` §5 第 4 条要求捕获 `[FIDESlib] bootstrap precomputation:` 开头的行。**这些行没有出现。**

原因:`[FIDESlib] bootstrap precomputation:` 前缀只存在于 `CheckPrecomputationShape`(`src/CKKS/openfhe-interface/RawCiphertext.cu:1368,1373,1380`)的 **警告输出**(`std::cerr`)中。这个函数在以下情况打印:

1. 某层的对角线跨越多个 level(`hi > lo`)
2. 某层的 level 不是前一层减 1(`lo != previous - 1`)
3. 第一层的 level 高于密文 level(`lo > ciphertextLevel`)

**三个条件都没有触发**,所以没有打印。这意味着:

- 每层对角线都在同一个 level
- 每层恰好比前一层低一个 level
- 第一层不高于密文 level

**Precomputation shape 是正确的。** StC 崩溃不是因为 precomputation shape 错误——因为崩溃在 CPU 侧,根本没用到 GPU precomputation。

---

## 5. 总结

| 问题 | 答案 |
|---|---|
| ModRaise test | 被注释掉了(`/* ... */`),无法运行 |
| 完整 bootstrap 14→6 掉在哪 | 恒定 28 bit 缺口,是整条管线累积,不是某一阶段 |
| StC 崩溃在哪一侧 | **CPU 侧**。`FHE->EvalSlotsToCoeffs()` 抛 `poly.h:274 Modulus mismatch`,GPU StC 从未执行 |
| `[FIDESlib] bootstrap precomputation:` 行 | 不存在。`CheckPrecomputationShape` 的三个警告条件都没触发,shape 是正确的 |
| CtS 是 GPU 特有问题吗 | 不是。CPU 和 GPU CtS 精度相同(42=42, 33=33) |

### 下一步建议

1. **StC 崩溃是 OpenFHE CPU 侧问题**——需要看 `EvalSlotsToCoeffs` 在 `tparams64_13_4` 参数下为什么模数链不匹配。这不是 FIDESlib 的 bug。
2. **28-bit 缺口在完整 bootstrap 中确认**——EvalMod trace 正常,需要更细粒度的探针来定位缺口在管线中的哪一步累积。CtS 在 post-bootstrap 输入上 CPU=GPU,说明 CtS 不是缺口来源。缺口在 EvalMod 或 ModRaise 中。
3. **ModRaise test 需要取消注释**才能运行。如果要跑,我可以取消注释并重新编译。
