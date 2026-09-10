# 回复 GPU-OOM-stage06-persists-20260909

日期：2026-09-10。本机没有 GPU，**C++ 改动一行都没编译过**。

「回收跑了但一块都没找到」这句话很关键——**它说明我上一轮的推理漏了一环**。补上之后，
这三个改动是相乘的关系，缺一个另外两个都白做。

先谢谢那个编译错误的修复：`ReclaimFreeSlabs` 用了 `s[]`，而 `FIDESlib::Stream s[MAXG]` 在它下面
才定义。我把函数移到了 `s` 的定义之后（比再写一个 extern 声明干净），你不用再本地打补丁。

---

## 一、我漏掉的那一环：释放密文并不会释放显存

`Ciphertext::~Ciphertext` 不 free，它把两个多项式**塞进 context 的辅助池**：

```cpp
Ciphertext::~Ciphertext() {
    if (!c1.GPU.empty()) cc.returnAuxilarPoly(std::move(c1));
    if (!c0.GPU.empty()) cc.returnAuxilarPoly(std::move(c0));
}
```

而 `trimAuxilarPoly` / `clearAuxilarPoly` **全项目没有任何地方调用**。所以那个池只涨不落，
稳定在「同时活着的密文数」的历史最高水位，然后一直占着。

于是链条是这样的，**三段缺一不可**：

```
密文析构 ──► 辅助池（我原来以为到这就完了）
辅助池 ──trim──► size class 空闲表
空闲表 ──整块空闲──► driver（上一轮加的回收）
```

**你看到的「回收跑了但没找到整块空闲的 slab」，正是因为中间那段断着**——所有块都还被辅助池攥着，
从空闲表的角度看它们全都「在用」。

## 二、补上中间那段

新增 `CryptoContextImpl::TrimAuxiliaryPolys(keep=0)` / `GetAuxiliaryPolyCount()`，绑到
`Engine.trim_auxiliary_polys()`，`Stages.release_pooled_memory()` 包一层（没有设备池的引擎上是 no-op，
所以 stage 代码可以无条件调用），**在 `EncoderLayer.forward` 的每个 stage 边界调用**。

选在 stage 边界是有讲究的：**stage 中途排空是负收益**——那些池化的多项式马上又要被用到，
排空只会让下一次分配把它们重新拿走。只有 stage 边界上工作集才是真的变小了。

## 三、顺着「85 分钟」这条线索又砍了两处

lazy alignment 把 13 分钟推到 85 分钟，说明方向对但峰值还是被什么撑着。找到了：

**`_accumulate_product` 的 `diagonals` 从头到尾全活着。** 每条只在一次迭代里用到，
但整个数组被调用方持有到函数返回。

* **stage 06**：`make_copies(q)` 是 **64 条**，depth=37 下约 **2.4 GiB**；
* **stage 08**：softmax 的 **128 条**，约 **4.9 GiB**——**这才是全层最大的工作集**。

加了 `consume` 参数：用完当场置 None。stage 06 无条件开（`make_copies` 的输出就地生成、别处不读）；
stage 08 在**不 trace 时**开（`--per-stage` 会在它返回后解密 softmax 输出，那时得留着）。

注意这**不会**把显存还给 driver（见第一节），但它让下一次迭代的密文**从辅助池里取**而不是
从堆顶新拿——高水位不再随循环往上爬。**加上第二节的 trim，这些块才真的能走完整条链。**

数值完全不变，有用例钉住「consume 与否结果逐位相同」。

---

## 四、下一次运行

参数不用变：

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
    --binary-rotations --refresh-after-dense --depth 37 --dnum 4 \
    --bootstrap-level-budget 3,3 --light-plaintext-cache 4
```

**建议先不加 `--per-stage`**：它会为了留 trace 而关掉 stage 08 的 `consume`，那是最大的一块。
先确认能不能跑完；跑通了再加 `--per-stage` 跑第二次拿逐级数据。

要看的：

* 有没有出现 `memory pool returned ... MiB of wholly free slabs to the driver`——
  **这一行现在应该会出现了**，上一轮没有正是因为中间那段断着；
* stage 06 / 08 过没过；
* 跑通的话请发回单层耗时、峰值显存、`key memory` 那一行。

如果还是卡在同一处，下一个能动的就是 `levelBudget={4,4}`（明文 7.6→约 6.9 GiB，但 depth 要 38），
以及把 `--light-plaintext-cache` 压到 1。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CudaUtils.cu` | `ReclaimFreeSlabs` 移到 `Stream s[]` 定义之后（你的编译错误） **未编译** |
| `api/CryptoContext.{hpp,cpp}`、`src/CKKS/Context.{cuh,cu}`、`python/src/bindings.cpp` | `TrimAuxiliaryPolys` / `GetAuxiliaryPolyCount` **未编译** |
| `python/pyfideslib/__init__.py` | `Engine.trim_auxiliary_polys` / `auxiliary_poly_count` |
| `python/thorfhe/stages.py` | `Stages.release_pooled_memory()`，无设备池时 no-op |
| `python/thorfhe/layer.py` | 每个 stage 边界调用；stage 08 在不 trace 时 `consume` |
| `python/thorfhe/attention.py` | `_accumulate_product(consume=)`，stage 06 无条件开 |
| `python/tests/test_gpu_resource_fixes.py` | 两个用例：consume 前后逐位相同、无池时 no-op |
