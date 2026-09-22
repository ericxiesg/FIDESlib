# 静态审查：`multiply/relinearize/rescale` 里**没有**任何 slot 0 特例；`automorph_slot` 是双射

日期：2026-09-22。设备静默期间做的源码审查，**结果全是否定的**——但否定得很有用：
它把"去 kernel 里找 index 0 分支"这条路关掉了。

---

## 0. 结论

| 查的东西 | 结果 |
|---|---|
| `multiply` / `relinearize` / `rescale` 路径上有 coefficient-0 / slot-0 特例吗 | ❌ **没有**。全是 `idx = threadIdx.x + blockDim.x*blockIdx.x` 的一致索引 |
| `idx == 0` 的分支 | 全仓只有 **printf 调试**，而且大部分还注释掉了（`Conv.cu:60/266/326` 在 `/* */` 里，`ElemenwiseBatchKernels.cu` 的带 `PRINT &&` 前缀） |
| `automorph_slot` 是双射吗（散射会不会重写/漏写某个槽） | ✅ **是双射**，穷举验证，65536/65536 |
| conjugate 把 slot 0 映到自己吗 | ❌ **不**。`0 -> 65535` |
| `addScalar` 负常数那段是 bug 吗 | ❌ **不是**（§3） |

---

## 1. `_times` 三个算子里没有 slot-0 机制

`multiply`（`binomialMult_`/`binomialMultExtend_`）、`relinearize`（`modup_ksk_moddown_mgpu`
→ `fusedDotKSK_2_`）、`rescale`（`rescale_fusion` / `SwitchModulus`）——
**每个 kernel 都用同一个线程索引一致地处理全部 N 个系数，没有任何 `i == 0` 分支。**

找到的 index-0 特例全部在 **limb / digit / tower 粒度**，例如：

- `LimbPartition.cu:1399` 的 `if (i == 0) Mult_ else addMult_`——digit 循环的累加器初始化；
- `NTT.cu:736` 的 `dat[0]`——rescale 模式下所有 block 读同一个源 limb；
- `LimbPartition.cu:695` 的 `limbsize - 1`——rescale 丢掉最高 tower。

**这些分支的判据对整个 block 是一致的，所以每个系数走同一条路。**
它们要是错了，**所有槽都错，不是 slot 0 错。**

> **所以：不要去 `multiply`/`relinearize`/`rescale` 里找 index 0 分支，没有。**

---

## 2. `automorph_slot` 是双射——散射是安全的

`src/Rotation.cuh:29` 的 `automorph__` 是一个**散射**：

```cpp
uint32_t rotIndex = automorph_slot(n_bits, index, j);
a_rot[rotIndex] = a[j];          // scatter
```

散射的风险很具体：**如果索引映射不是双射，某个槽会被写两次、另一个槽一次都没写**
（保留上一轮的陈旧数据）。那正好是"单槽损坏"的形状，所以值得查。

`automorph_slot` 是纯整数运算，我在本机把 logN=16 的全部 65536 个槽穷举了一遍：

| index | 用途 | 不同像的个数 | 判定 | `slot 0 ->` |
|---|---|---|---|---|
| 131071 | **conjugate**（2N−1） | 65536/65536 | **双射** | **65535** |
| 5 | rotate 1 | 65536/65536 | 双射 | 16384 |
| 25 | rotate 2 | 65536/65536 | 双射 | 12288 |
| 52429 | rotate −1 | 65536/65536 | 双射 | 26214 |
| 122881 | rotate 2048 | 65536/65536 | 双射 | 15 |
| 28609 | rotate 16 | 65536/65536 | 双射 | 2028 |

**全部双射，没有重写也没有漏写。这条关掉。**

### 2.1 顺带：slot 0 **不是**共轭的不动点，连实现层面也不是

`0 -> 65535`。我在 `efc57a3` 里说过"slot 0 是共轭自同构的不动点"，
在 `4837ca8` 里以"共轭在槽上原地作用"为由撤回了。现在实现层面也验证了一遍：
**即使按这份实现的槽置换来看，slot 0 也不是不动点。** 这条线彻底关了。

---

## 3. 一个看着像 bug 但不是的东西

`Ciphertext.cpp:947` 的负常数分支：

```cpp
if (c < 0.0) {
    for (auto i = 0u; i < elem.size(); ++i)
        elem[i] = cc.prime[i].p - elem[i];      // elem[i]==0 时得到 p，未约简
}
```

看上去危险：`p` 没有约简到 `[0, p)`，而 `AddSub.cu:107` 的 `scalar_add_` 调 `modadd`。
**但 `modadd` 正好能吃下它**（`AddSub.cuh:37`）：

```cpp
T tmp0 = a + b;
return (tmp0 >= prime_p) ? tmp0 - prime_p : tmp0;
```

`a + p >= p` 恒成立 → 返回 `a + p - p = a`，正是"加 0"的正确结果；
而 `a + p < 2p < 2^61`，64 位不溢出。**不是 bug，不用改。**

（写在这里是因为它看起来确实像一个，下次再有人扫到这行可以省一次追查。）

---

## 4. 这对定位意味着什么

`38dbf00` 把范围缩到了 `_times` 的三个算子，而本次审查说**那三个算子里没有单槽机制**。
两件事都成立的话，剩下的可能是：

1. **损坏不是"某个算子对 slot 0 特殊"，而是数据相关**——同一个 kernel 对某个特定的
   输入值算错（比如某个模约简的边界），而 slot 0 恰好是那个值所在的位置；
2. 或者 `iter01_b` 的 slot 0 在更早就已经有偏差，只是没进 `inv_input_lift3` 的 worst-3
   （worst-3 只说谁最大，不说谁健康）。

**两条都由已经推上去的那两个实验直接分辨，而且都不需要再读源码：**

- `THORFHE_SWAP_TIMES=1` —— 别名 vs 操作数；
- `test_times_at_slot_zero_with_the_iterations_own_operands` —— 三个算子里的哪一个，
  而且它同时报 `b*correction` 和 `ones*correction`，第二组正好检验"数据相关"这条。

**审查到此为止。** 再往下读源码的边际收益已经低于跑那两条命令。
