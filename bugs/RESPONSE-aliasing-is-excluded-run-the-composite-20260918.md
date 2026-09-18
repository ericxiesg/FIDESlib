# 别名不成立（三处证据），下一步跑组合测试 + 看 noise_level

日期：2026-09-18。针对 `63dda1c`。

---

## 0. 别名假设可以排除

你怀疑 `correction` 在一轮里被用两次，第一次乘法把它就地改了。查过了，三处都挡住：

**1. `multNoRelin` 的参数是 const**（`src/CKKS/Ciphertext.cpp:751`）：

```cpp
void Ciphertext::multNoRelin(const Ciphertext& b) {
```

右操作数改不了。`correction` 在 `_times(a, correction)` 和 `_times(b, correction)` 里**都是右操作数**
（`Stages.multiply` 只在左边不是密文时才交换，这里两个都是密文，所以不换）。

**2. `adjustForMult` 那条路我们根本不走**（同文件 756 行）：

```cpp
if (cc.rescaleTechnique == FIXEDAUTO || FLEXIBLEAUTO || FLEXIBLEAUTOEXT) {
    if (!adjustForMult(b)) { ... }
}
```

我们是 **FIXEDMANUAL**，整段跳过。

**3. 左操作数也没有别名**。`EvalMultNoRelin` 里 `make_shared<CiphertextImpl>(*ct1)` 走拷贝构造
（`api/Ciphertext.cpp:25`），它调 `CopyDeviceCiphertext`（`api/CryptoContext.cpp:1956`）：

```cpp
auto new_ct = std::make_shared<FIDESlib::CKKS::Ciphertext>(context_gpu);
new_ct->copy(*ct_gpu);
uint32_t handle = this->RegisterDeviceCiphertext(std::move(new_ct));
```

**新对象、新 handle、深拷贝。**

顺带：`prepare_for_multiply` 在 FIXEDMANUAL 下是 `return x`，也不是嫌疑。

---

## 1. 我这边又排除了一条：correction 上的绝对误差

你测出三个算子在**绝对**量级 1e-10~1e-12 干净。我想过这可能仍然不够：
iter03 的 correction 是 `0.0158389 − b`，而 b 自己最大就是 0.0164——两者贴在一起的那些 slot，
correction 本身接近 0，1e-10 的绝对误差在那里就是 100% 的相对误差。

**注入了，不放大：**

| 给 `subtract` 注入的绝对误差 | 输出最差相对误差 |
|---|---|
| 0 | 4.873e-05 |
| 1e-10 | 4.871e-05 |
| 1e-8 | 5.3e-05 |
| 1e-6 | 8.2e-04 |

线性透传。Goldschmidt 在这一点上是良态的。

---

## 2. 现在的排除表

```
✅ variance 灾难性相消        ✅ 自举越界（余量 24.5 倍）
✅ ones 空槽泄漏（真实存在，×1.158e4，但只出现在 a，且需要 1e-2 量级）
✅ 分母空槽泄漏（无影响）      ✅ 承载槽相对噪声（线性）
✅ correction 绝对误差（线性） ✅ 别名（const + 深拷贝）
✅ prepare_for_multiply（恒等）✅ 旋转密钥计划（21 把全覆盖）
✅ 三个算子孤立测试（你测的）
```

**ClearEngine 在我能构造的任何扰动下都产生不出 219.6。** 单个算子也都是干净的。
剩下的只能是**组合**或**状态**。

---

## 3. 两件事，都很便宜

### 3.1 跑已推的组合测试

`python/tests/test_he_inv_primitives.py`（`a8a3719`）：

```
cd python && PYFIDESLIB_DEVICES=cpu,cuda:0 python -m pytest tests/test_he_inv_primitives.py -q
```

和你的孤立测试的区别就是它**不孤立**：最后一个用例把整步 Goldschmidt 连做 5 步，
用 `he_inv` 真实的标量序列，断言**量级不增长**。每个算子都在容差内而组合发散，
正是"0.0039 → 219.6 然后平方"的形状，孤立测试按定义看不到。

另外前三个用例是**带 level 的**（drop 0/4/8/10）——你的孤立测试如果是在满 level 上做的，
那低 level 还没覆盖，而迭代正是在低 level 跑的。

### 3.2 把 noise_level 打出来

`GetNoiseLevel` 已经绑定了，是个便宜的调用。在 `he_inv` 每轮打一次：

```python
print(step, engine.noise_level(a.ciphertext), engine.noise_level(b.ciphertext))
```

FIXEDMANUAL 下**每轮结束都必须是 1**。如果哪一轮变成 2，那后面每一步都在错误的 Δ 上算，
结果会是"值不对"而不是"值更吵"——正好是我们看到的形状。

值得怀疑是因为这台设备**已经有过一次 noise_level 异常**：
`EvalBootstrap` 在 FIXEDMANUAL 下返回 scale degree 2（`Engine.bootstrap` 里那个 rescale 循环
就是为它加的）。而 `he_inv` 的第一步就是 `bootstrap(denominator)`。

还有 `multNoRelin` 里那两行：

```cpp
assert(NoiseLevel == 1);
assert(NoiseLevel == b.NoiseLevel);
```

**Release 构建里 `assert` 是空的。** 如果 noise_level 真的漂了，这两道本该拦住的检查不会响。
请确认你的构建是不是 `-DNDEBUG`；如果是，用 Debug 或 `-UNDEBUG` 跑一次 `he_inv`，
这两个断言会直接告诉我们答案。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
