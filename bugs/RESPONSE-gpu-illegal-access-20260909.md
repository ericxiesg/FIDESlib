# 回复 GPU-illegal-memory-access-20260908

日期：2026-09-09。本机没有 GPU，**C++ 改动一行都没编译过**。

先说进展：**depth=34 / dnum=4 / (3,3) / refresh + binary rotations 这组的预算是对的**——你实测
23,172 MiB、余量 4.6 GiB，和模型预测的 27.4 GiB / 4.6 GiB 一致，而且 **stage 01–06 跑完了，
第一次自举真的开始执行了**。这是目前走得最远的一次。

然后：**slab 分配器不是这次的原因，我的改动也不太可能是。** 但你的第 3 点是真 bug，而且我找到了
确切机制。逐条如下。

---

## 一、`GPUfree` 不做任何指针运算，混合 slab 大小与它无关

`GPUfree` 在 cache 路径上的全部工作是：

```cpp
std::vector<void*>& free_limb = size_to_memory[id][bytes];
s[id].wait(stream);
mempool_lock[id].lock();
free_limb.emplace_back(ptr);      // ← 就这一句
mempool_lock[id].unlock();
return;
```

没有 slab 基址、没有偏移计算、没有链表遍历、没有读块内元数据。空闲表就是一个
`std::vector<void*>`，装的是裸指针——**一个指针来自 1 GiB 的 slab 还是 128 MiB 的 slab，
对它完全没有区别**。所以你的假设 1 和 3 不成立。

假设 2（竞态）也不太可能：`emplace_back` 是在 `mempool_lock` 里做的。

## 二、真正的原因：`cudaErrorIllegalAddress` 是**异步**的，`GPUfree` 只是第一个报告它的地方

kernel 里的非法访存不会在 launch 处报错，它是**粘滞错误**，要等到下一次同步/检查才浮出来。
`GPUfree` 里恰好有

```cpp
s[id].wait(stream);
CudaCheckErrorModNoSync;      // ← 报错报在这里
```

所以崩溃点是 `GPUfree`，**故障点是它之前某个 kernel**。堆栈里 `~LimbPartition → GPUfree` 只说明
「析构时同步了一下，把之前的错捞出来了」。

结合堆栈，出问题的 kernel 在 **`multMonomial` 内部**——这个函数以前就出过事（MAXP 溢出那次，
`EvalMultByI → multMonomial` 的 NaN/segfault 就是它第一个撞上被污染的常数）。它是自举里
唯一大量调用的、会现场构造临时 `RNSPoly` 的原语。

### 一次就能定位的办法

```bash
CUDA_LAUNCH_BLOCKING=1 python -m thorfhe.bench fhe ...        # 报错点变成真正的 kernel
compute-sanitizer --tool memcheck python -m thorfhe.bench fhe ...   # 直接给出越界的 kernel 和地址
```

**这比继续猜有效得多，也是我最希望你先做的一件事。** 我这边没 GPU，只能给出最可疑的一处（见下）。

## 三、最可疑的一处，已加断言

`Ciphertext::multMonomial` 现场构造单项式时：

```cpp
RNSPoly monomial(cc.getAuxilarPoly());
monomial.grow(c0.getLevel());
monomial.dropToLevel(c0.getLevel());
...
int limb_size = g.getLimbSize(monomial.getLevel());
for (int i = 0; i < limb_size; ++i)
    SWITCH(g.limb[i], load(coefs));        // ← 用 level 推出的个数去索引 limb 向量
```

循环上界来自 **level**，而索引的是 `g.limb` **向量**。`RNSPoly::grow` 有一句
`if (level >= new_level) return;`——从池子里拿到的 `RNSPoly` 若 `level` 字段已经够高，
就**直接返回、不分配**。只要 `limb.size()` 和 level 不一致，`g.limb[i]` 就越界，
拿到的是垃圾 `LimbImpl`，kernel 于是写到非法地址——**然后在几次调用之后的某个同步点才报出来，
正好符合你看到的现象**。

已加检查：两个循环前都验 `limb_size <= g.limb.size()`，不满足就抛带两个数字的异常。
如果这就是原因，下次跑会得到一句明确的错误而不是 illegal address；如果不是，这句检查也不会误伤。

## 四、你的第 3 点是真 bug，机制找到了：`std::map::emplace` 静默丢弃

```cpp
precom.keys.at(ksk.keyID).rot_keys.emplace(index, std::move(ksk));
```

**`std::map::emplace` 在 key 已存在时什么都不做**，并且**不会**替换。

同一个旋转索引会被请求两次：一次来自 `SetRotationKeyLevels`（你的电路在哪些 level 旋转），
一次来自自举预计算（StC / CtS 在哪些 level 旋转）。**binary rotations 下两边都是 2 的幂，
必然重叠。**

于是：先加进去的是**你的计划里那把截断过的键**，自举后来要加的完整键被**静默丢弃**，
自举再在更高的 level 上用它——`ensureLevel` 抛错。这就是你看到的「level 33 用了截到 28 的键」，
而且**你的计划没有错**，是这里把两个需求合并错了。

已改：索引重复时**保留覆盖更高 level 的那把**（完整键 `maxLevel < 0` 胜过任何截断键）。
你那个「所有键设成 level 34」的 workaround 就不再需要了，那 356 MiB 也能省回来。

## 五、关于「revert 自适应 slab」

不建议。它做的事只是「1 GiB 要不到就减半重试」，而且这次 **它显然起作用了**——上一轮你在
`GPUmalloc` 就崩了，这一轮走到了第一次自举。真正该做的还是**让空 slab 回到 driver**
（记录 slab 基址 + 全空检测），那是根治；自适应只是让它在碎片化时别硬失败。

不过如果 compute-sanitizer 指向了我改的那段，请立刻告诉我，我马上回退。

---

## 六、下一次运行建议

按这个顺序，每步都能单独给出结论：

1. **先 `compute-sanitizer --tool memcheck`**（或 `CUDA_LAUNCH_BLOCKING=1`）跑同一组参数——
   直接拿到越界的 kernel。这一步最值钱。
2. 拉这版重跑，**不要**再用「所有键设 level 34」的 workaround：第四节那个 bug 修了之后，
   正常的截断计划应该就能用了。
3. 顺便验证一下第五节新加的**自举键逐层计划**：跑一次带 `--allow-key-grow`，
   key memory 报告里的 "grown at runtime" 应该是 **0**。是 0 就说明逐层模型对。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CKKS/Context.cu` | `AddRotationKey` 索引重复时保留覆盖更高 level 的键，不再静默丢弃 **未编译** |
| `src/CKKS/Ciphertext.cpp` | `multMonomial` 两个 limb 循环前加边界检查，越界变成带数字的异常 **未编译** |
