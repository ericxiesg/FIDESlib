# GPU 的算术是精确的，所以那 27 bit 根本不是"精度"问题

2026-09-23。补 `cc698cf` 里提的三个候选（NTT、rescale/moddown、key switch precision），以及 Part23 §6 的候选 4（sm_70 上的 128-bit 累加）。

---

## 0. 结论

**生产路径上的模乘全是精确整数运算，没有任何浮点。** 所以"累加精度丢失"这个机制在这里不存在，那 27 bit 不可能是精度问题。

---

## 1. Part23 的候选 4：确实有 fp64 路径，但没被用

Part23 §6 #4 怀疑："`MakeAuxPlaintext` 在 CPU 侧用 `int128_t`；GPU 侧 multPt/累加若走 64-bit 快路径，误差形状恰好会是确定性、与消息无关、随槽位变化。"

这个怀疑是有实据的——`ModMult.cuh` 里**确实有** fp64 路径：

```cpp
// ModMult.cuh:178
const uint64_t rx = fp64himult(op1 << (53 - qbit), op2 << ...);
```

`53 - qbit` 就是 FP64 的尾数位。50~55 bit 的素数在 53 bit 尾数下确实会掉精度。在 V100（sm_70，FP64 1:2）上选它也完全合理。

**但它没有被选。**

```cpp
// src/CKKS/forwardDefs.cuh:34-36
enum ALGO { ALGO_NATIVE = 0, ALGO_NONE = 1, ALGO_SHOUP = 3, ALGO_BARRETT = 4, ALGO_BARRETT_FP64 = 5 };
constexpr ALGO DEFAULT_ALGO = ALGO_BARRETT;
```

`ALGO_BARRETT_FP64` 在整个树里只出现在 `bench/LimbNTTBenchmarks.cu`、`test/LimbKernelTest.cu`、`test/NTTtests.cu`——**`src/` 下一次都没有**。

- 元素级乘法：`DEFAULT_ALGO = ALGO_BARRETT`，整数 Barrett，用 `__umul64hi` + `__uint128_t`（`ModMult.cuh:47,61`）
- NTT / INTT：`Limb.cuh:78,80` 默认 `ALGO_SHOUP`，Shoup 乘法，预计算 `mu` + `__umul64hi`，同样是精确整数

候选 4 退场。

---

## 2. 由此收紧的，不只是候选 4

RNS-CKKS 在 device 上就是**模素数的精确整数运算**。这不是"精度很高"，是**根本没有舍入**。于是：

**你们列的三个候选里，两个的机制不成立。**

- "Rescale/moddown precision —— GPU rescale 舍入方式不同，误差累积 37 层"：rescale 确实是唯一带舍入的运算（除法取整），但每次贡献是 **1 ulp 量级**。37 层攒不出 27 bit。
- "Key switch precision"：key switch 全程精确整数，没有"precision"可言。
- "NTT/INTT correctness"：注意这条你们写的是 *correctness* 不是 precision，这是对的——但一个错的 NTT（错 twiddle、错根）给出的是**乱数**，不是"降级 27 bit 的答案"。而且现有 `OpenFHEInterfaceTests` 在 logN 16 / depth 23 上是绿的。

所以 27 bit 只能来自这三类之一：

1. **舍入次数/位置不同**——但这是 1 ulp 级，除非某处 rescale 多做或少做了**整数次**，那就不是精度而是 scale 错位
2. **算法本身不同**——`ApproxModEval.cu` 是 FIDESlib **自己实现**的，不是从 OpenFHE 搬的
3. **某个常数差了 2 的幂**

而 **σ = 0.0156 = 2⁻⁶ 精确等于 2 的幂**。精确整数算术里攒出来的误差不会落在 2 的整数幂上；**差一次 2 的幂次缩放会**。第 2、3 类都指向同一个地方。

---

## 3. 所以请不要动这三个内核

`a3bef1d` 里我就说了先别动 NTT / key switch / rescale，现在有了更具体的理由：**它们的机制根本解释不了 27 bit**。在四行分段 `Max error` 回来之前改通用路径，是在没有定位的情况下动全局，风险远大于收益——而且这三条路径在别的配置下是被测试覆盖且绿的。

---

## 4. 还是那四行

重建之后：

```
OpenFHEBootstrapTests/OpenFHEBootstrapTest.ModRaise        (stage 1)
OpenFHEBootstrapTests/OpenFHEBootstrapTest.CoeffsToSlots   (stage 2)
OpenFHEBootstrapTests/OpenFHEBootstrapTest.ApproxModEval   (stage 3 —— 首选嫌疑)
OpenFHEBootstrapTests/OpenFHEBootstrapTest.SlotsToCoeffs   (stage 4)
```

每个都是 GPU 对 OpenFHE CPU 同一密文、按 slot 取 max，`22ef07f` 之后带 `tparams64_16_thor_fixmanual`（FIXEDMANUAL / depth 37 / scale 50 / sparse ternary）。把新那组和原来八组的 `Max error` 行一起贴回来。

同一次运行顺手带上（都是免费的）：

- `FIDESLIB_TRACE_MODEVAL=1` 的三行 scale degree（`f1c2717`）。预期进 double-angle 是 degree 2、出来也是 2；**进去是 1 就是发现**。
- 启动日志里 `[FIDESlib] bootstrap precomputation:` 开头的行。按 Part23 §3.3 的结构分析应当一行都没有。

---

## 5. 另外：softmax 逐步对比已经可用

`d75e15d`。`python/tests/test_softmax_step_by_step.py`：

```
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_softmax_step_by_step.py
```

同一组 scores 在 device 和精确算术上各跑一遍 `stage_07_softmax`，按探针名对齐——打开 `THORFHE_DEBUG` 后一个 stage 07 有 **72 个探针**，包含 `he_inv` 里每一次 Goldschmidt 的 `a`/`b`。输出四张表：device-vs-clear、device 各步从自身实测输入重算、clear 同关系基线、`|b|` 双列对照。

**每一步从"实测的上一步"重算**，而不是拿端到端理论值比——后者让每一步为上游背锅，整条尾巴全红，第一个真故障和它自己的后果分不开。局部重算问的是"给定实际进来的输入，这步输出对不对"，第一个过不了的就是坏的那步。

在精确算术上的基线（已实测，不是猜的）：`07c` 从 `07b` 重算残差 **8.7e-19**，`07d × 07c` 跨 slot 常数性 **6.8e-4**（就是 Goldschmidt 残差）。device 列偏离这两个数就是定位点。

测试数据用的是 **MRPC validation[0] 经真 checkpoint 算出来的分数**，不是合成数——`Softmax.NARROW` 的窗口 `[-27.25, 21.73]` 是在真实 MRPC 激活上标定的，合成分数两端都碰不到，拿它比设备会把标定问题混进 FHE 误差。
