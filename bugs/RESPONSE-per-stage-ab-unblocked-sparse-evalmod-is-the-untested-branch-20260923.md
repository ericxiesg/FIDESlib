# 分段 A/B 解锁了，不用 depth 37；真正没测过的是 sparse EvalMod

2026-09-23，针对 `66ac9fe`。代码在 `9c2bdae`。

## 0. 先更正我自己

我在 `report-cts-stc-review-20260923.md` 和两条 commit message 里写过："TTALL64BOOT 八组参数全是 FIXEDAUTO 或 FLEXIBLEAUTOEXT，FIXEDMANUAL 从来没跑过。"

**这是错的。** `tparams64_13_2_fix` / `_3_fix` / `_4_fix` 都是 **FIXEDMANUAL**，logN 16、dnum 最高到 4，而且它们是绿的。只有 `_13_1_fix` 是 FIXEDAUTO——我读了那一个就推广到八个，和之前 `dropLimb`、`dropToLevel` 两次是同一类错误（拿截断的证据当完整的）。

所以 **FIXEDMANUAL 是覆盖的，CtS/StC 在它下面（depth 23）是对的。**

## 1. 这反而把范围缩小了

对照 benchmark，现在没测到的差异只剩三个：depth（23 vs 37）、模数（59/60 vs 50/55）、**密钥分布**。

第三个是要害，而且确实一次都没测过——所有参数集都硬编码 `UNIFORM_TERNARY`。密钥分布决定 EvalMod 的近似：

| | Chebyshev | double-angle |
|---|---|---|
| sparse | `g_coefficientsSparse` | `R_SPARSE = 3` |
| uniform | `g_coefficientsUniform` | `R_UNIFORM = 6` |

而 **`ApproxModEval.cu` 是 bootstrap 里唯一由 FIDESlib 自己实现、不是从 OpenFHE 搬运的一段**。benchmark 跑的正是它没被任何测试进入过的那个分支。

## 2. 解锁：不需要 depth 37

`tparams64_13_4_sparse` —— depth 23、dnum 4、FIXEDMANUAL、SPARSE_TERNARY。**和旁边那些集合一样大，不额外吃显存**，所以 depth 37 装不下的那个对比，现在能跑了。

depth 37 那组保留，挪到 `FIDESLIB_TEST_THOR_DEPTH37` 后面，默认不编进去。**所以直接重建就行，不用加任何定义**，OOM 不会再发生。

## 3. 一个坑：光设 SetSecretKeyDist 没用，而且更糟

`GetRawParams(cc, boot_conf)` 的第二个参数**默认 UNIFORM，且不从 context 推导**（`RawCiphertext.cuh:108`）。

所以只在 OpenFHE 侧设 `SPARSE_TERNARY` 的话，会变成：**OpenFHE 按 sparse 造 bootstrap 预计算，FIDESlib 用 uniform 系数和 6 次 double-angle（而不是 3 次）去求值。** 这种错配读起来不像配置错误，读起来就是精度 bug——正是我们在找的东西。

`GeneralParametrizedTest::bootConfig()` 现在从参数集推导它，36 个 `GetRawParams` 调用点全部改为传入。对现有 UNIFORM_TERNARY 的集合返回 UNIFORM，**行为不变**。

## 4. 请跑

```
OpenFHEBootstrapTests/OpenFHEBootstrapTest.ModRaise
OpenFHEBootstrapTests/OpenFHEBootstrapTest.ApproxModEval      <- 首选嫌疑
OpenFHEBootstrapTests/OpenFHEBootstrapTest.CoeffsToSlots
OpenFHEBootstrapTests/OpenFHEBootstrapTest.SlotsToCoeffs
```

把 `tparams64_13_4_sparse` 那一组的 `Max error` 行，连同原来八组的一起贴回来。**如果 sparse 那组红而八组全绿，27 bit 就定位到 `ApproxModEval.cu` 的 sparse 分支了。**

顺带（同一次运行，免费）：`FIDESLIB_TRACE_MODEVAL=1` 的三行、`[FIDESlib] bootstrap precomputation:` 开头的行（按 Part23 §3.3 应当一行都没有）。

## 5. 你们剩余嫌疑里我要减两条

`66ac9fe` 列的四条里：

- **"ModRaise 的 `generateSpecialLimbs(false, true)` 可能保留陈旧数据"**——`broadcastLimb0_` 的 guard 恒真（`adaptTo` 给所有素数打 U64），每个 limb 都被写。另外注意：你们的确定性测试是**同一进程内重复调用**，分配器模式相同，所以"确定但陈旧"的缓冲区能躲过它——这条我不敢说完全排除，但它解释不了 27 bit 这个量级。
- **"key switch 截断引入额外噪声"**——测试用的是全量未截断密钥（这正是 OOM 的原因），而 CPU 33 bit / GPU 6 bit 的对比就是在 `bench_params()` 下测的，两边截断设置相同。而且 key switch 是精确整数运算，见 `59f4d6c`。

`evalChebyshevSeries` 的数值那条我同意，而且 `ApproxModEval` 测试正好覆盖它——这也是为什么第 4 节那四行值得优先。
