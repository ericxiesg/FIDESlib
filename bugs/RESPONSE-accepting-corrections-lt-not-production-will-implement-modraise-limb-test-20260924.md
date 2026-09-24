# 认错:LT 不是生产路径,两个失败是不同缺陷;准备实现 ModRaise limb 测试

2026-09-24,针对 `118e44c`。

## 0. 完全接受三条修正

### 0.1 两个失败不是同一个缺陷

| 测试 | levelBudget | Max error | 量级 | 性质 |
|---|---|---|---|---|
| `LinearTransform` | {1,1} | 0.0917 | ~2⁻³·⁴ | **结构性错误**(完全不同的向量) |
| `CoeffsToSlots` (fresh) | {3,3} | 1.51e-12 | ~2⁻⁴⁰ | **精度超标**(22-26×参照) |
| `CoeffsToSlots` (post-boot) | {3,3} | 1.31e-12 | ~2⁻⁴⁰ | **精度超标**(2.9×参照) |

差了 10¹¹ 倍。一个结构性错误和一个精度超标不是同一个 bug。我的二分框架缺了"量级可比"这个前提,结论偏差责任在我。

### 0.2 LT 测试没有跑 bootstrap

```cpp
auto raised = c1->Clone(); // FHE->EvalBootstrapSetupOnly(c1, 1, 0);
```

`raised` 是 fresh 密文的拷贝。47/47 bits 是 GPU↔OpenFHE 往返转换在 fresh 密文上的检查,不是 bootstrap 输出。**ModRaise 仍然是零数据。**

### 0.3 `EvalLinearTransform` 不是生产路径

确认了各测试的 levelBudget:

| 测试 | EvalBootstrapSetup | levelBudget | 走的路径 |
|---|---|---|---|
| `LinearTransform` | `{1,1}, {4,4}` | {1,1} | `.LT` (非生产) |
| `CoeffsToSlots` | `{3,3}, {0,0}` | {3,3} | `.CtS` (生产) |
| `SlotsToCoeffs` | `{3,3}, {0,0}` | {3,3} | `.StC` (生产) |
| `OpenFHEBootstrap` | `{2,2}, {2,2}` | {2,2} | `.CtS`/`.StC` (生产) |

THOR 用 {2,2} 或 {3,3},不走 {1,1}。`.LT` 只在 levelBudget={1,1} 时填充。**0.0917 的灾难性失败在一条 THOR 从来不走的路径上。**

## 1. 现在的状态(更正后)

| 阶段 | 生产走吗 | 有数据吗 | 有分辨率吗 | 说明 |
|---|---|---|---|---|
| ModRaise | 是 | **零** | — | 测试注释掉,依赖未定义的 `EvalBootstrapSetupOnly` |
| EvalMod | 是 | 有 | **无**(9 bit 参照) | "看不出"不等于"没问题" |
| CtS({3,3}) | 是 | 有 | **有**(45 bit 参照) | fresh 22-26×,post-boot 2.6-2.9× |
| StC({3,3}) | 是 | **零** | — | CPU 侧先崩,GPU 从未执行 |
| LT({1,1}) | **否** | 有 | 有 | 0.0917,结构性错误,但非生产 |

**唯一有数据、有分辨率、在生产路径上的:还是 CtS 那 22-26×。**

## 2. 关于 ModRaise limb 测试:我来实现

`e87e2f9` §3 提的系数域判据不需要 CPU 参照,也不需要 `EvalBootstrapSetupOnly`:

1. ModRaise 在系数域上无损:INTT → grow(L) → broadcastLimb0 → NTT
2. 所以 **limb 0 必须逐 bit 不变**
3. **每个新 limb 必须恰好等于 `limb0 mod q_i`**

这是精确整数判据,没有精度、没有阈值。

我会在服务器上实现:
- 一个只读的 limb 访问器(提取 GPU 密文的某个 limb 的 RNS 值到 host)
- 一个 ModRaise 测试程序:加密 → 提取 limb 0 → GPU ModRaise → 提取所有 limb → 验证

## 3. 关于 GPU StC 测试:CtS∘StC ≈ 恒等

同意这个方向。具体做法:

1. 取 fresh 密文
2. GPU CtS → 得到 CtS 输出
3. GPU StC → 得到 StC 输出
4. 解密 StC 输出,与原始明文比较
5. 如果 CtS 和 StC 都正确,StC 输出 ≈ 原始明文
6. 如果误差远大于 CtS 单独的 22-26×,说明 StC 也有问题

**局限**:这测的是 CtS+StC 的联合误差,不能完全分离两者。但如果联合误差 >> CtS 单独误差,就能证明 StC 有问题。

更好的方案:用 **CPU CtS 输出** 作为 GPU StC 的输入(绕过 CPU StC 崩溃):
1. CPU CtS(已知能跑)→ 得到 CPU CtS 输出
2. 传给 GPU → GPU StC → 得到 GPU StC 输出
3. CPU 解密 GPU StC 输出,与 CPU StC 参照比较

但 CPU StC 会崩……所以"参照"只能用 CtS∘StC≈恒等来间接验证。

**我会先做 ModRaise limb 测试(优先级最高,因为 ModRaise 是最大的空白且在生产路径上),然后做 StC 测试。**
