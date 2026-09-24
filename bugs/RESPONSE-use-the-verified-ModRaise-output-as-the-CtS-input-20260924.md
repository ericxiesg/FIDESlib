# Im 半边测不了是我的输入选错了——用已经验证过的 ModRaise 输出去喂它

2026-09-24,针对 `b3bbe14`。

## 0. 测试是对的,Re 半边也复现了

`b48b9d4` §4 预测 "Re 那一半应该复现现有测试的 20.9×"——实测 **20.8×,36 bit 参照**。构造是对的,折叠也照 `Bootstrap.cu:119-130` 复现了。

## 1. Im 参照只有 3 bit,这是我提的输入的问题

我在 `b48b9d4` §4 写的是:

> 用现有测试的输入(`x1 = {0.25, 0.5, ...}`,8 个值)以保持参照精度

**那个输入只保住了 Re 半边的参照。** 8 个非零值稀释进 32768 个 slot,系数向量的幅度很小,折叠出来的 Im 半边小到 `GetLogPrecision` 只报 3 bit——于是 Im 那一列又变成了噪声比噪声,和 `2f0e338` 同一个毛病,只是这次是我造成的。

## 2. 最好的修法:拿**已经被精确验证过的 GPU ModRaise** 的输出当输入

生产里 CtS 拿到的不是 fresh 密文,而是 **ModRaise 之后**的密文——系数是 `Δm + e + q0·I`,**稠密、幅度大、两个半边都饱满**。这正是 Im 半边不退化成噪声所需要的输入。

而 `96b0fa4` 已经把 GPU ModRaise **按精确整数判据验证过了**(limb0 逐 bit 相同,每个新 limb = `centered(limb0) mod q_i`,两组参数都过)。**它是整个调查里唯一被精确验证的一段。**

所以这条路是干净的:

1. 加密 → GPU `ModRaise`(已验证正确)
2. 把结果搬回 OpenFHE 一次,作为 **CPU 和 GPU 共同的输入**
3. CPU `EvalCoeffsToSlots` + 折叠 → `re_cpu` / `im_cpu`
4. GPU `EvalCoeffsToSlots` + 折叠 → `re_gpu` / `im_gpu`
5. 两个半边分别比

**不需要 `EvalBootstrapSetupOnly`**(那个未定义的函数),因为 GPU ModRaise 替代了它,而且它的正确性不是假设,是测出来的。

## 3. 如果上面嫌麻烦,先试这个一行改动

把输入从 8 个值改成**填满所有 32768 个 slot**的随机实向量:

```cpp
std::vector<double> x1(slots);
std::mt19937 rng(23);
std::uniform_real_distribution<double> d(-1.0, 1.0);
for (int i = 0; i < slots; i++) x1[i] = d(rng);
```

稠密输入让系数向量的幅度上来,两个半边都不会掉到噪声下。**这只是一行,先跑它看 Im 参照能不能从 3 bit 抬到 30+ bit**;能抬上去就说明 §2 不是必需的。

## 4. 为什么值得继续追 Im 半边

不是因为我认定它有问题,而是因为**现在的证据结构**:

| | 参照分辨率 | 结果 |
|---|---|---|
| CtS `2·Re` | 36 bit | **20.8× 超标** |
| CtS `2·Im` | 3 bit | 测不出 |
| StC(复合) | — | 相对 CtS 干净 |
| EvalMod | 9 bit | 测不出 |
| ModRaise | 精确整数 | **清白** |

生产同时消费两个半边(`ctxt+conj` 和 `ctxt−conj` 都喂给 `approxModReduction`)。**Re 半边已知超标 20.8×,而 Im 半边一次都没被有效测量过。** 在 28 bit 还没有着落的情况下,一个从没被测过、又被生产使用的量,值得先量出来再说。
