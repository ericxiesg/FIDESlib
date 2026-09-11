# 回复：bootstrap 无罪，`he_exp` 炸了——给你一个几秒钟的复现，别再用 25 分钟的基准调它

日期：2026-09-11。**C++ 本轮未改。**

## 一、探针把范围又缩了一半，也推翻了我的猜测

```
07a.refreshed_scores   [-10.43, +12.48]     ✅ 正确
07b.exp                ~1e+124              ❌
```

**bootstrap 是好的。** 我上一份说「通往 07a 的路径上只有 bootstrap 没验证过，所以它是首要嫌疑」——
那句话的**前提是 07a 已经坏了**，而 07a 没坏。这是我第三次在这个 bug 上猜错方向
（`EvalNegate`、噪声底噪、bootstrap），三次都是探针/对照把我纠回来的。
**继续用数据推进，不要用我的猜测推进。**

## 二、为什么偏偏是 `he_exp`：它是唯一走这条路的地方

stage 01–06 精确到 3e-7，但它们乘的是**掩码明文**和**整数**。
`evaluate_polynomial` 是这个移植里唯一一处同时做这三件事的代码：

1. **浮点标量乘**（`multiply(ct, float)` → `EvalMultScalar` → `multScalar(double)`）；
2. **在 degree-2 密文上做上一条**——`power_basis` 的 `basis[k] = rescale(square(...))`，
   `square` 默认 `relin=False`，所以基元素是 degree 2；
3. 把这些项 `align` 到同一 level 再相加，常数项用 `addScalar` 加进去。

`test_stage2_linear.py` 里的 `test_mult_float_scalar_consumes_level` 覆盖了**单次**浮点标量乘加
rescale，而且在 GPU 上是过的。**所以问题在组合，不在单个算子。**

## 三、新增 `tests/test_polynomial_reproducer.py`：秒级复现

五个检查，**每个只隔离一个原语**，哪个挂了就说明去看哪儿：

| 检查 | 隔离什么 |
|---|---|
| `power_basis` | `x^2/x^3/x^4/x^8`，惰性重线性化的幂基 |
| `float_scalar_on_degree_two` | 浮点标量乘**一个未重线性化的密文**（`multScalar` 要不要处理第三分量） |
| `scalar_added_to_degree_two` | 常数加到 degree-2 密文（`addScalar` 只动 `c0`） |
| `small_polynomial` | 7 次，最小的带 giant step 的形状 |
| `exp_polynomial` | **THOR 的 15 次 exp 拟合本身，输入就是探针实测的区间除以 32** |

每个检查跑两遍：一遍对精确算术（证明检查本身没写错），一遍对设备（要问的那个）。
两半独立 skip，所以没编扩展的机器上也能用。本机精确那半 **5/5 通过**——
**说明移植的多项式逻辑在精确算术下是对的**，问题确实在引擎侧。

```bash
python3 -m pytest tests/test_polynomial_reproducer.py -q
```

**如果 `exp_polynomial[device]` 挂了，你就有了一个几秒钟的循环，不用再跑 25 分钟。**

## 四、如果上面五个在 GPU 上全过

那说明故障依赖 **level 或模数链长度**——上面的检查跑在 SMALL（log_n 13、depth 12），
而基准里 `he_exp` 跑在 log_n 16、depth 37、密文在 level 18 附近。

所以另加了一个 opt-in 检查，**用基准的真实参数**建引擎，并把密文先降到 level 18 再求值：

```bash
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest \
    tests/test_polynomial_reproducer.py::test_exp_polynomial_at_benchmark_parameters -q
```

建密钥要几分钟，但仍然远少于 25 分钟，而且它只跑这一个算子。

## 五、建议的顺序

1. 先跑那五个（秒级）；
2. 全过就跑 `PYFIDESLIB_BENCH_PARAMS=1` 那个（分钟级）；
3. 两者都过，才值得再跑完整基准——那说明是 `he_softmax` 里 `he_exp` **周围**的东西
   （`normalised` 的那次 `multiply(ct, 1/32)`、`level_down(merged, 3)` 之后的 level，
   或者 `he_exp` 里的 `add(x, -shift)`），下一轮我把探针再往里加一层。

上一轮那三处 `multScalar(-1.0)` 的修复照旧要编译——它们是真 bug，只是**不是这个病**
（`he_inv` 在 `07d`，而 `07b` 先坏）。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/tests/test_polynomial_reproducer.py` | 新增。五个隔离检查 ×（精确 / 设备），外加一个基准参数下的 opt-in 检查 |
