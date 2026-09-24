# 那个 spread=101 是"噪声除以噪声";但你们的直觉指向一个**真的从没测过的半边**

2026-09-24,针对 `8a915a8` / `2f0e338`。

## 0. 先更正一条事实

> The CPU side does conjugate+add (doubles real part, cancels imaginary), **GPU doesn't (for dense)**

**不对,现有的 `CoeffsToSlots` 测试两边都做了 conjugate+add。** 在 `test/OpenFheInterfaceTests.cu`,以 `TEST_P(OpenFHEBootstrapTest, CoeffsToSlots)` 起点(`:3470`)计:

```
+114  auto conj = FHE->Conjugate(ctxtEnc, evalKeyMap);   ← CPU 侧
+115  cc->EvalAddInPlace(ctxtEnc, conj);

+146  auto conj = FHE->Conjugate(cResGPU, evalKeyMap);   ← GPU 侧,同样做
+147  cc->EvalAddInPlace(cResGPU, conj);
```

所以"conjugate 不匹配"不能解释那 20.9×。

## 1. 决定性的问题:你们新测试的参照只有 **4 bit**

你们自己输出里的这一行:

```
Max error: 0.165079 (Expected: 0.125)
```

`ASSERT_ERROR_OK` 里 `Expected = 2^(-GetLogPrecision()+1)`。`0.125 = 2^-3` ⇒ **`logPrecision = 4`**。

**CPU 参照自己只有 4 bit 精度,误差棒是 0.125。**

而你们列的 CPU 值:

```
slot0: CPU= 0.0175      slot1: CPU= 0.0110      slot3: CPU=-0.0637
slot4: CPU= 0.0597      slot7: CPU=-0.0163
```

**全部比 0.125 这个误差棒还小。** 也就是说:**这些 CPU"参照值"本身就是噪声。**

拿噪声去除噪声,得到的正是 `min=-44.6, max=19.8, spread=101` 外加符号翻转。**这个形状不是"结构性错误"的证据,它是"两个都在噪声底下"的证据。**

对照一下现有测试:`Expected = 2.91e-11 = 2^-35` ⇒ **`logPrecision = 36`**。差了 32 bit 的参照分辨率。

**所以 `2f0e338` 分辨不出"GPU 算错了"和"CPU 参照在这个配置下没有意义"。**

## 2. 为什么去掉 conjugate 会把参照打到 4 bit

**生产环境在 CtS 之后立刻就做这个折叠**(`Bootstrap.cu:119-130`,dense 分支):

```cpp
EvalCoeffsToSlots(ctxt, slots, false);

if (cc.N / 2 == slots) {            // ← 生产走这里
    aux.conjugate(ctxt);
    Ciphertext ctxtEncI(cc_);
    ctxtEncI.sub(ctxt, aux);        // ctxt − conj  →  2i·Im
    ctxt.add(aux);                  // ctxt + conj  →  2·Re
    ctxtEncI.multMonomial(3 * 2 * cc.N / 4);
    ...
    approxModReduction(ctxt, ctxtEncI, ...);
}
```

**CtS 折叠前的那个复数值是内部表示,不是 CtS 被契约要求产出的量。** 生产一次都没用过它。去掉折叠去比它,比的是一个两边约定可以合法不同的中间量——参照精度塌到 4 bit 就是这么来的。

## 3. **但你们的直觉指向一个真的空白,而且我之前也没注意到**

看上面那段:生产用了**两个半边**——`ctxt + conj` 的 `2·Re`,**和** `ctxt − conj` 的 `2·Im`。

而现有的 `CoeffsToSlots` 测试只做 `EvalAddInPlace(ctxtEnc, conj)`,**只测了 `2·Re` 那一半**。

**`2·Im` 那一半,到今天为止一个测试都没有碰过。**

这很要紧,因为一个"在 `+conj` 下抵消、在 `−conj` 下翻符号"的约定差异:

| | CPU `z` vs GPU `conj(z)` |
|---|---|
| `z + conj(z)` | **相同** ← 现有测试测的,所以绿 |
| `z − conj(z)` | **符号相反** ← 生产用的,没人测过 |

**这种缺陷会对现有每一个测试隐形,同时在生产里污染 `ctxtEncI`。** 而 `ctxtEncI` 正是喂给 `approxModReduction` 的第二个参数。

(注:你们贴的数不支持"GPU = conj(CPU)"这个具体形式——若是,实部会相等而它们不等。但第 1 节说明那些数本来就在噪声下,所以既不能证实也不能证伪。)

## 4. 该做的测试:复现生产的折叠,**两半都比**

不是去掉折叠,而是**照 `Bootstrap.cu:119-130` 原样做**,然后 CPU / GPU 各自的两个半边分别比:

1. CPU:`EvalCoeffsToSlots` → `conj` → `re_cpu = ctxt + conj`,`im_cpu = ctxt − conj`(再 `multMonomial`)
2. GPU:同样的序列
3. **分别比 `re_cpu vs re_gpu` 和 `im_cpu vs im_gpu`**

`re` 那一半应该复现现有测试的 20.9×(参照 36 bit,有意义)。**`im` 那一半是新信息。** 如果 `im` 显著更差,那就是 28 bit 的落点,而且解释了为什么所有现有测试都看不见它。

用现有测试的输入(`x1 = {0.25, 0.5, ...}`,8 个值)以保持参照精度,别用会把 logPrecision 打到 4 的配置。

## 5. 状态

| 阶段 | 状态 |
|---|---|
| ModRaise | 清白(精确整数判据) |
| CtS `2·Re` 半边 | **红着**,20.9×,36 bit 参照,生产路径 |
| CtS `2·Im` 半边 | **从未测过** ← 新 |
| StC | 相对 CtS 干净;dense 配置已验,但只采样了 0.8% 的 slot(`a316b98`) |
| EvalMod | 有数据,无分辨率(9 bit 参照) |

`2f0e338` 那个 spread=101 我不认为是结论,但**它把注意力引到 conjugate 折叠上,而折叠的另一半确实是个真空白**——这个功劳算你们的。
