# 交换实验是决定性的，别名排除了；但"`_times` 单独跑是干净的"这一条**不算数**——我的测试跑错了 level

日期：2026-09-23。回应 `11b1379`。

---

## 0. 结论

| | |
|---|---|
| 别名 | ✅ **排除**，干净利落。换序后坏槽三个位置**一个不变**（0, 9891, 20170） |
| `_restore_magnitude` 空操作 | ✅ 你验证了，逐位相同 |
| "`_times` 单独跑在 slot 0 没问题" | ⚠️ **这条不成立**——**我的测试在 level 36 跑，迭代在 level 24 跑**（§2） |
| 你 §4.3 的猜想（bootstrap 后的密文不一样） | ✅ **对，而且我一起漏了**——测试两个操作数都是 fresh `encrypt` |
| 已修 | 测试现在跑四种：top level / 降到迭代 level / **bootstrap 过的 b** / ones 对照 |

---

## 1. 交换实验：别名排除，而且排除得很干净

| | no-swap `b_times` | swap `b_times` |
|---|---|---|
| worst at | **0(%0), 9891(%3), 20170(%10)** | **0(%0), 9891(%3), 20170(%10)** |

**三个槽一模一样，顺序都一样。** 别名的话坏槽会跟着第二次调用跑到 `a`，它没有。

而 `a_times` 在 swap 后反而干净了（`>1 0/32768`），worst 变成 436/12724——
**正是分母最小的那一族**，健康时本来就该在那里（我在 `4837ca8` 里给过这个参考）。

**所以：不是别名，是操作数决定的，而且是确定性的**（两次跑同样三个槽）。这条很硬。

---

## 2. 但"`_times` 单独跑干净"这一条要撤回——是我的测试写错了

你的 test 输出里有这一行：

```
b * correction  (level 36):
```

**而迭代里的乘法发生在 level 24 → rescale 到 23。**（本机参考：
`inv_input_lift3` level 24，`inv_iter01_b` level 23。）

> **我的测试在剩 36 个 limb 的时候做乘法，迭代在剩 24 个的时候做。**
> **一个"limb 少了才出现"的故障，那个测试根本看不见。**

更难看的是：同一个文件里另外三个测试都带 `drop` 参数、都在 0/4/8/10 个 level 下跑过
（`_at(engine, ct, drop)` 就是我为这件事写的），**偏偏最关键的这个我没用**。

### 2.1 你 §4.3 的猜想是对的，而且我一起漏了

你说：test 里 `b` 是 `engine.encrypt(b0)`，而迭代里 `b` 是 **bootstrap 的输出**，
`correction` 是从那个 bootstrap 密文减出来的。

**对。** 这是第二个 gap，和 level 是独立的两件事：bootstrap 出来的密文，
它的 NoiseLevel、scale、以及内部的 limb 结构，都不必然和 fresh encrypt 一样。

---

## 3. 已修：四种情形

```python
cases = [
  ("fresh, top level",              encrypt(b0),            encrypt(correction)),   # 原来那个，已知干净
  ("fresh, dropped to level 24",    _at(encrypt(b0), 12),   _at(encrypt(correction), 12)),
  ("bootstrapped b",                bootstrap(encrypt(b0)), subtract(2/k, <同一个密文>)),  # ← 忠实复现
  ("ones * correction",             encrypt(ones),          encrypt(correction)),   # 对照
]
```

判读：

| 哪一行先坏 | 说明什么 |
|---|---|
| **fresh, dropped** 坏 | **是 level**：limb 少了之后 multiply/relinearize/rescale 出问题 |
| 只有 **bootstrapped** 坏 | **是密文的来源**：bootstrap 产出的密文结构和 fresh encrypt 不同 |
| 两个都干净 | `_times` 真的没问题，那就剩"`correction` 是从 `b` 自己减出来的"这一层相关性——
  迭代里两个操作数**不独立**，而测试里它们是独立的 |

最后那一行值得单独说：迭代里 `correction = 2/k − b`，所以
**`b × correction = b(2/k − b)`，两个因子是同一个密文的函数**。
我的测试用两个独立加密的向量去乘，即使数值一样，**密文层面的相关性没有复现**。
如果前两行都干净，这就是下一个要复现的东西（而且很容易：
用同一个 `b` 密文减出 correction，就是上面第三行在做的）。

---

## 4. 请跑

```bash
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q \
    tests/test_he_inv_primitives.py::test_times_at_slot_zero_with_the_iterations_own_operands
```

四行都会打印 slot 0 对其余 32767 个槽的 p50 的倍数，以及产品所在的 level。
**`-s` 不能省。**

---

## 5. 顺带一个观察

你 §1.3 提到 no-swap 的 `a_times` 有 `>1 1/32768` 在 31280(%0)，swap 后消失。

那个大概率是正常的：`a_times = ones × correction = correction`（在 carried 槽上），
而 `correction = 2/k − b₀ = 1.0469 − b₀`。b₀ 的下端在 0.045 附近时 correction 就到 1.002，
**越过 1 是算术上应该的，不是故障**——`>1` 只是探针的一个阈值，对 `correction` 没有意义。

真正不该越界的是 `b_times`：它的上确界是 `1/k² = 0.274`（`b₀(2/k − b₀)` 的顶点），
而设备报 1.505，**5.5 倍**。这条才是要查的。
