# 扫一遍：我们哪些改动会影响"不是我们"的工作负载

日期：2026-09-14。起因是 bert-tiny 那个 crash——
`alignToDiagonals` 是为 FIXEDMANUAL 写的，却无条件作用在 FLEXIBLEAUTO 上。

那不是一个孤立失误，是**一类**：我们为 THOR / FIXEDMANUAL 加的东西，落在了所有调用者身上。
上次 `multScalar(-1.0)` 我只修了 `EvalNegate` 没扫全，被远程抓到；这次先扫完再说。

判据：**main 上不存在、bootstrap-dev 上存在，且对一个既不用 FIXEDMANUAL、也不是 THOR 的调用者会改变行为。**

---

## 一、已修

| 改动 | 问题 | 处理 |
|---|---|---|
| `CoeffsToSlots.cu` 的 `alignToDiagonals` | 为 FIXEDMANUAL 的差一 bug 写的，无条件执行；FLEXIBLEAUTO 下多降 level，降到 -1 | 已加 `rescaleTechnique == FIXEDMANUAL` 前置判断（`acb256f`） |
| `Ciphertext.cpp` 的 `addPt`/`subPt` scale 守卫 | —— | 一开始就写了 `rescaleTechnique == FIXEDMANUAL`，其它技术保持原 assert ✓ |

## 二、仍然无条件、但我认为应当保留

| 改动 | 为什么无条件是对的 |
|---|---|
| `ConstantsGPU.cu` 的 `L + K <= MAXP` 检查 | 这是**常量表的硬约束**，和 scaling technique 无关。超了就是越界写，任何技术下都是 bug |
| `LimbPartitionBatch.cu` 的 `LTdotProductPtBatch` limb 检查 | 越界读在任何技术下都是 bug。main 没有这个检查而能跑，说明 FLEXIBLEAUTO 下不存在不匹配——**那检查就是无害的**；如果存在，main 是在静默越界读，**那检查就是有价值的**。两种情况下都该留 |
| `Ciphertext.cpp` 的 `multMonomial` limb 检查 | 同上 |

> 但有一条要写下来：**这两个检查会把 main 上"静默错但能跑完"的情况变成"抛异常跑不完"。**
> 如果远程在别的工作负载上撞到它们，那不是回归，是检查在做它该做的事——
> 报错文本里有两个 limb 数，直接说明是哪儿对不齐。

## 三、仍然无条件，而且**我认为有风险**

### 3.1 `truncate_keys` 默认为 `true` —— **建议改成 `false`**

```cpp
bool truncate_keys = true;        // api/CryptoContext.hpp:311
bool truncateKeys  = true;        // src/CKKS/Context.cuh:182
```

level 截断密钥是我们为"单卡放不下 250 把满 level 旋转键"加的。它**默认开**，
所以**任何**用 FIDESlib 的人都会拿到截断过的密钥和一张为他们工作负载算的 level 计划——
而那张计划的正确性我们只在 THOR 上验证过。

bert-tiny 的日志证明它确实在生效：

```
[FIDESlib] bootstrap key level plan: 46 of 61 keys truncated (L=25, bootstrap depth 20, ...)
```

计划算错的后果不是崩，是 `ensureLevel` 抛异常（`allow_key_grow=false` 时）
或者悄悄重载（开了 grow 时，代价是每次调用一次 host->device 传输）。

**建议**：默认改成 `false`，让需要的人显式打开（`thorfhe.bench` 传 `truncate_keys=True`）。
理由和上游 PR 草稿第 5 节写的是同一条——**这是对既有用户的行为变更**，
而它的收益只有在密钥放不下时才存在。**我没有直接改**，因为这会影响远程正在跑的 THOR 实验，
需要同时改 `bench` 的默认值，建议一起决定。

### 3.2 `EvalSubInPlace(double scalar, ct)` 的符号我改过

原来算 `ct - scalar`，我改成 `scalar - ct`（`9bdd560`），理由是：
它的 CPU 回退委托给 `EvalSubInPlace(scalar, ctImpl)`，而紧邻的另一个重载
`EvalSubInPlace(ct, scalar)` 已经是 `ct - scalar`——两个参数顺序相反的重载算出同一个东西，
必有一错。

**但这是本分支唯一一处我主动改了语义的地方**，而且我当时就写了"请用 OpenFHE 源码核实"。
bert-tiny 不调它（查过），THOR 移植也不调，所以目前没有已知调用方——
**但如果有第三方调用方，我们改了它的算术。** 上游 PR 之前必须核实。

### 3.3 `light_plaintext_cache_capacity = 64` 默认非零

缓存 64 个展开后的明文。对不用 light plaintext 的调用者是死代码（不会有条目），
所以风险低，但它是一块默认占用的显存。列在这里只为完整。

---

## 四、一句话

`alignToDiagonals` 那个 bug 的教训不是"我漏了一个判断"，是
**"为自己的场景加的东西，默认作用在所有人身上"**。
上面第三节的两条（截断密钥默认开、`EvalSubInPlace` 改符号）是同一形状，
只是还没被撞到。提上游之前必须各自有个决定。
