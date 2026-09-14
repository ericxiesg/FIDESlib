# 回复 `c04c142`：那个测试测不到它想测的东西——把字段直接暴露出来

日期：2026-09-14。**C++ 改动未编译。**

## 一、`addScalar` 是 scale-自适应的，所以那个断言恒真

你的推理是：

> 如果 bootstrap 把 `NoiseLevel` 留在 N > 1，`addScalar(1.0)` 把常数按 Δ 编码，
> 而密文在 Δ^N，所以加出来会差 Δ^(N-1)。

前提不成立。`Ciphertext::addScalar` 把 `this->NoiseLevel` 传了进去：

```cpp
auto elem = cc.ElemForEvalAddOrSub(c0.getLevel(), std::abs(c), this->NoiseLevel);
```

而 `ElemForEvalAddOrSub`（`Context.cu:478`）用它把常数**额外乘 `noise_deg - 1` 次
scaling factor**：

```cpp
for (uint32_t i = 1; i < static_cast<uint32_t>(noise_deg); i++) {
    crtConstant = CKKSPackedEncoding::CRTMult(crtConstant, crtScFactor, moduli);
}
```

常数被编码成配套的尺度，**加出来是对的**——不管 `NoiseLevel` 是 1 还是 2。
所以那个断言在两种情况下都成立，**它不可能因为它想检测的原因而失败**。
测不到的测试比没有测试更糟，因为通过会给出错误的信心。

（顺带纠正我自己：上一轮扫 `multScalar` 时我写过「`addScalar` 按当前 NoiseLevel 编码常数，
写法正确」——那句话是对的，但我当时没想到它的**副作用**是让这类探测失效。）

一个边角：`logApprox > 0` 那条分支在应用 `noise_deg` 之前就 `return` 了。
`Δ = 2^50` 下这要 `|常数| > 2^12 ≈ 4096` 才会走到，我们的常数都是 O(1)，碰不到。

## 二、与其推断，不如把字段读出来

`NoiseLevel` 是 FIXEDMANUAL 下密文状态的一半，而**此前没有任何地方能读它**。
这个项目已经为此花掉好几轮：取负耗不耗一个 degree、bootstrap 回来是不是 canonical，
全靠从下游的错误答案往回推。它其实就是一个字段。

加了 `GetNoiseLevel`，一路到 Python：

```
CryptoContextImpl::GetNoiseLevel(ct)   # GPU 读 Ciphertext::NoiseLevel；CPU 读 GetNoiseScaleDeg()
  -> bindings.cpp  GetNoiseLevel
  -> Engine.noise_level(ct)
  -> ClearEngine.noise_level(ct)   # 返回 scale_exp，两个引擎同一个接口
```

## 三、重写后的测试

`tests/test_bootstrap_noise_level.py` 现在用两条独立的证据：

1. **直接读**：`assert engine.noise_level(ct) == 1`；
2. **走后果**：`engine.add(ct, np.ones(slots))` —— **明文永远编码在 Δ^1**，
   所以如果密文不在 Δ^1，上一轮加的 `addPt` FIXEDMANUAL 守卫会**抛异常**，
   而不是返回一个看着合理的错数。

另外加了一个在普通 SMALL 参数下就能跑的用例，钉住访问器本身的语义：

```python
assert engine.noise_level(x) == 1
assert engine.noise_level(engine.multiply(x, 0.5)) == 2      # 浮点标量乘耗一个 degree
assert engine.noise_level(engine.rescale(...)) == 1          # rescale 还回来
```

**这个用例很重要**：它让 `noise_level` 自己有回归保护，否则将来它悄悄返回常数 1
也不会有人发现——那正是你那个测试遇到的问题的另一种形态。

## 四、下一跑

重编之后先跑这两个（秒级 + 分钟级），再跑基准：

```bash
python3 -m pytest tests/test_polynomial_reproducer.py -q
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest tests/test_bootstrap_noise_level.py -q
```

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `api/CryptoContext.{hpp,cpp}` | `GetNoiseLevel` **未编译** |
| `python/src/bindings.cpp` | 绑定 **未编译** |
| `python/pyfideslib/__init__.py` | `Engine.noise_level` |
| `python/thorfhe/clear.py` | `ClearEngine.noise_level`，两个引擎接口一致 |
| `python/tests/test_bootstrap_noise_level.py` | 重写：直接读字段 + 用守卫走后果；另加访问器自身的回归 |
