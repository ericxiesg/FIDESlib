# `broadcastLimb0_` 的 ISU64 guard 是真 bug，但在我们的配置下打不开；下一个该试的是 correction factor

2026-09-23，针对 `4fec711`。

---

## 0. 结论

你们读的内核和 guard **完全属实**，`ElemenwiseBatchKernels.cu:140-148` 就是那样写的，`ISU64(x) = constants.type & (1 << x)`（`ConstantsGPU.cuh:187`）。作为一个 latent bug 它值得修。

**但它在我们这个配置下不可能触发**，所以它不是 0.0156 的成因。

---

## 1. 为什么打不开

你们把"是否有 uint32 素数"列为待验证项。答案是：**没有，而且不是因为位宽，是因为被显式写死了**。

`Context.cu:262` 的类型选择是

```cpp
.type = (prime[i].type ? *(prime[i].type) : (prime[i].bits <= 30 ? U32 : U64))
```

`prime[i].type` 是一个 optional 覆盖。位宽回退（`bits <= 30`）只在它为空时才用。而 Python `Engine` 这条路上它**从来不为空**：

```cpp
// api/CryptoContext.cpp:226
params = params.adaptTo(rawParams);
```

```cpp
// src/CKKS/Parameters.cu:12-18
for (auto i : raw.moduli)
    new_primes.push_back(PrimeRecord{ .p = i, .type = U64 });
for (auto i : raw.SPECIALmoduli)
    new_SPECIALprimes.push_back(PrimeRecord{ .p = i, .type = U64 });
```

**`adaptTo` 给每一个 Q 素数和每一个 special 素数都显式打上 `U64`。** 所以 `constants.type` 全 1，`ISU64(primeid) && ISU64(0)` 恒真，guard 那条 else 分支走不到，`broadcastLimb0_` 对每个 limb 都写。

顺带一提，即使没有 `adaptTo` 这一手，位宽回退也救得了我们：`first_mod_bits=55`、`scaling_bits=50`，`bits = std::bit_width(p)`（`Context.cu:203`）算出来是 50~56，没有任何一个 ≤ 30。**两道保险都在。**

## 1.1 仍然建议修

guard 本身是错的：它在 `ISU64(primeid) != ISU64(0)` 时静默不写，把未初始化显存留在密文里。今天靠 `adaptTo` 写死 U64 挡住了，但那是两个相隔很远的文件之间的隐含约定，没有任何断言维系。`broadcastLimb0_mgpu_`（:153）同病。

最小修法不是补 uint32 分支（那条路今天没人走，写了也测不到），而是**让它响**：

```cpp
if (ISU64(primeid) && ISU64(0)) { ... }
else { __trap(); }   // 或 assert，宁可炸也不要留未初始化 limb
```

---

## 2. 关于"非确定性"：请先跑 `test_stage_stability_one_ciphertext.py`

`b4ffc8a` 里那份报告说明了为什么 `556efcd` / `104f27d` 的测量不成立——两个测试都把 `engine.encrypt(b0)` 写在循环里，而 ModRaise 解密出的是 `Δm + e + q0·I`，`I` 跟着随机的 `a` 走。**换加密必然换结果，这是正确 ModRaise 的定义。**

这跟本报告 §1 是独立的两件事，但结论方向一致：目前**没有证据**表明 device 上存在非确定性。

`test_stage_stability_one_ciphertext.py` 是判定它的测试：加密一次，四个 stage 各跑四遍，**按位相等**断言。如果 `broadcastLimb0_` 真的留下了未初始化显存，这个测试会直接红——它恰好是为这种 bug 设计的。所以请把它跑了再决定要不要动内核：

```
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_stage_stability_one_ciphertext.py
```

绿 = 非确定性这条线关闭，`grow` / `generateSpecialLimbs` / `Accumulate` / `broadcastLimb0` 全部洗清。

---

## 3. 下一个该试的：correction factor

协作方给了一份对 CtS/StC 编码的独立核查（对照 OpenFHE 1.5.1 上游、deps 补丁、EasyFHE 的逐函数移植），结论是 **FIDESlib 根本不"编码"CtS/StC，只是把 OpenFHE 算好的 `m_U0hatTPreFFT` / `m_U0PreFFT` 原样搬到 GPU**。据此排掉了一批候选：

- 层内对角线 limb 不齐 → `multPt` 截断：**结构上不可能**。同一层 s 的 63 条对角线共用同一个 `paramsVector[s]`，limb 数严格相同；层间逐层 erase 一个 Q，正好对上运行时每层一次 mod-down。（这也从机制上解释了 09-15 为什么那条告警从未出现。）
- CtS/StC 层序颠倒：`RawCiphertext.cu:1153` 的 `std::reverse(result.CtS...)` 与 `AddBootstrapPlaintexts` 的倒序导入是配套的，没有错位。
- 对角线数值算错：FIDESlib 不算。
- 预计算按 sparse / 运行时按 uniform 的 k 错配：链路是通的。

**剩下的头号候选是 correction factor。**

`EvalBootstrapSetup(..., correctionFactor=0)` 让 OpenFHE 按拟合曲线自己选，而那条曲线**分支是按 scaling technique 分别拟合的**，`ckks-bootstrapping-precision.cpp` 标定的是 FLEXIBLE* 那支，我们走的 FIXEDMANUAL 是另一支：

```
round(-0.1516 * 47 + 14.284) = 7,  clamp[6,13] -> 7
```

而 thor-openfhe 自己的 `full_bootstrap_probe.cpp` 显式传 **12**。

correction factor 是 ModRaise 前后的 2^k 放大量，**是 bootstrap 精度的一阶项**。我们比参考实现低了 5，而实测精度是 10.9 bit（σ = 0.0156 = 32·2⁻¹¹），比预期低 10 bit 上下。这是目前最便宜、最可能的解释。

已经把它做成 `Engine` 参数（之前写死 0）：

```python
pf.Engine(device, ..., bootstrap_correction_factor=12)
```

`python/tests/test_bootstrap_correction_factor.py` 扫 `0, 7, 9, 10, 11, 12, 13`，每个值建一个 engine、bootstrap 一个已知向量、报 max / rms / 等效 bit：

```
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_bootstrap_correction_factor.py
```

**不设阈值，要的是曲线形状。** 如果精度对 k 平坦，这个候选就关掉；如果朝 12 单调改善，我们同时拿到了成因和修法。

---

## 4. 其余仍开着的候选（按便宜程度）

1. **correction factor**（上面，最便宜）
2. **EvalMod 段**。`ApproxModEval.cu` 是 FIDESlib **自己实现**的，不是搬运——sparse 档用 degree-44 Chebyshev + `R_SPARSE = 3` 次 double-angle。这是 `stopAfterStage` 本来就该分段定位的地方。
3. **StC 出口的 scale degree / rescale 次数**。`forwardDefs.cuh:40 MODRAISE_WITH_P0 = false` ⇒ `multScalar(constantEvalMult)` 之后不 rescale，密文以 degree 2 进 CtS，由 `CoeffsToSlots.cu` 开头补掉。净效果应当等价，但这正是 `72dc818` 那族问题的上游。
4. **sm_70 的 128-bit 累加路径**。`MakeAuxPlaintext` 在 CPU 侧用 `int128_t`；GPU 侧 `multPt` / 累加若走 64-bit 快路径，误差形状恰好会是"确定性、与消息无关、随槽位变化、幅度固定"——**和我们实测的形状一致**。这条需要读 kernel 确认，还没查。

---

## 5. 一并请求的数据

1. `test_stage_stability_one_ciphertext.py` 的结果（§2）——决定非确定性这条线的死活。
2. `test_bootstrap_correction_factor.py` 的七行输出（§3）。
3. 重建后 `OpenFHEBootstrapTests/OpenFHEBootstrapTest.CoeffsToSlots` 和 `.SlotsToCoeffs` 的 `Max error` 行，特别是新加的 `tparams64_16_thor_fixmanual` 那组——这是第一次在 FIXEDMANUAL + depth 37 + sparse ternary 下测 CtS/StC。
4. 启动日志里 `[FIDESlib] bootstrap precomputation:` 开头的行。按上面的分析**应当一行都没有**；出现了就是真发现。
