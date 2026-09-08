# 回复 GPU-OOM-runtime-grow-20260908 + fidelity-gpu-fideslib-20260908

日期：2026-09-08。本机没有 GPU，**下面的 C++ 改动一行都没编译过**。

先说好消息：`fidelity-gpu-fideslib-20260908` 是**第一次在真实 GPU FHE 上验证 stage 01–05**，
query relRMSE **1.03e-8**、best-fit scale 1.0000，而且 64 次旋转真的是用 15 把二进制密钥做出来的。
这一份报告顶掉了很多不确定性：**引擎原语、light plaintext、binary rotations、pcmm 在硬件上都是对的**，
剩下没验证的只有依赖 bootstrap 的部分。

然后是那个 runtime OOM——**找到了，是内存池，不是 `grow`**。

---

## 一、根因：`GPUmalloc` 是按「精确字节大小」分类的 slab 池，slab 从不还给 driver

`src/CudaUtils.cu`：

```cpp
uint64_t MBs = 1024;                                  // ← 默认 slab = 1 GiB
...
std::vector<void*>& free_limb = size_to_memory[id][bytes];   // ← 按 *精确字节数* 分类
if (free_limb.empty()) {
    uint64_t* base;
    cudaMallocAsync(&base, MBs * 1024 * 1024, s[id].ptr());   // ← 返回值没检查
    for (uint32_t i = 0; i < MBs * 1024 * 1024; i += bytes)
        free_limb.emplace_back(((char*)base) + i);
}
```

而 `GPUfree` 在 `cache` 路径上只是把块放回**它自己那个 size class 的空闲表**：

```cpp
free_limb.emplace_back(ptr);
return;          // ← 没有任何路径把 slab 还给 driver
```

所以：

* 任何 ≥ 64 KiB 的分配（limb 在 N=2^16 下是 512 KiB）**只要它那个 size class 的空闲表是空的，
  就会去要一整块 1 GiB**。
* 释放不会把内存还给 driver，只会还给**那一个** size class。
* 于是跑到后面，"free" 的显存大部分躺在**别的** size class 的空闲表里，谁也用不了。

这解释了报告里每一个现象：

| 现象 | 解释 |
|---|---|
| 8,434 MiB 空闲却 OOM | 要的不是 16 MiB，是**一整个 1 GiB slab**，而真实连续空闲不足 1 GiB |
| 崩在**第一次**计算 | 第一次密文拷贝要的 limb size class 是 keygen 从没碰过的，空闲表当然是空的 |
| `--light-plaintext-cache 2` 没用 | 释放的明文块回到**明文的** size class，limb 那个 class 一点也拿不到 |
| `--lenient` 没用 | 那是 Python 侧的 level 检查，和显存无关 |

## 二、逐条回答你的四个问题

**Q1「`generateLimbToLevel` 分配多少？」**
它自己只要 `N * 8` = 512 KiB 一个 limb，一个 depth-51 密文约 33 MiB。**但它经由 `GPUmalloc` 触发的是
1 GiB。** 这就是差了两个数量级的地方。

**Q2「为什么 `copy` 会触发 `grow`？」**
正常行为，不是 bug。`RNSPoly::copy` 是

```cpp
this->dropToLevel(poly.level);
this->grow(poly.level);
```

目标 `RNSPoly` 来自 `cc.getAuxilarPoly()`，池空时是个**全新的空 poly**，所以要先长到源的 level。
一次拷贝长出 33 MiB 是对的。

**Q3「grow 的临时内存能不能及时释放？」**
它不是临时的——`Ciphertext::~Ciphertext` 已经把 `c0`/`c1` 还给了 `returnAuxilarPoly`。真正的问题是
**还回去的内存也没离开进程**：池化的 RNSPoly 继续持有它的 GPU 块，就算 `trimAuxilarPoly` 被调用
（顺带一提，它和 `clearAuxilarPoly` **定义了但全项目没有任何地方调用**），块也只是回到 size class
空闲表，还是不会还给 driver。所以「及时释放」在当前分配器下拿不到收益。

**Q4「不截断密钥会不会好一些？」**
**不会，只会更糟。** 不截断的密钥更大（18,900 vs 17,452 MiB，+1,448 MiB）。而且你自己的日志里写着
`0 grown at runtime`——**从来没有任何密钥在运行时增长过**。堆栈里那个 `grow` 是**密文**拷贝，不是密钥。
所以我没有给 `fhe` 加 `--no-truncate`：它在这里是纯负收益。

---

## 三、已改（未编译）

`src/CudaUtils.cu`：slab 要不到就**减半重试**，一直到单块大小；全都失败才抛异常，并把
`cudaMemGetInfo` 的真实空闲量和「池按精确大小分类且不归还」这件事写进错误信息。顺带检查了
`cudaMallocAsync` 的返回值（之前没检查），并把切分循环的边界从 `i < slab` 收紧成
`i + bytes <= slab`。

**这很可能直接解掉你这次的崩溃**：见下一节，depth=51/(4,4) 只差 0.8 GiB，而失败的动作是「要一整个
1 GiB」。改成能退到 128 MiB 之后，大概率就过去了。

---

## 四、你的 (4,4) 实测已经进模型了

谢谢，这组数正是模型缺的。加进 `MEASURED_BOOTSTRAP` 之后：

| | 你的实测 | 模型 |
|---|---:|---:|
| bootstrap 明文 | 6,882 MiB | 6,882 MiB |
| 密钥常驻 + 明文 | 23.8 GiB | **23.8 GiB** |
| + "everything else" 9 GiB | — | 32.8 GiB |
| 32 GiB 卡余量 | — | **−0.8 GiB** |

也就是说 **depth=51/dnum=4/(4,4) 本来就装不下，差 0.8 GiB**——和你观察到的「keygen 过了、
第一步计算就崩」完全吻合。

顺便：我现在知道那 9 GiB **是什么**了。它不是什么神秘开销，就是**密文工作集**（光 stage 02 就同时
持有 64 份旋转副本）**按 size class 向上取整到整数个 slab**，加上那个从不排空的辅助 poly 池。
模型里的注释已经改成这个说法。

`bootstrap_depth` 也记进去了：**(3,3) 是 14，(4,4) 是 18**。所以 (4,4) 把 level 墙从 depth≥52
推到了 depth≥56——你实测 depth=50/(4,4) 过不了 level 墙（bl=32），正是这个。

---

## 五、接下来

1. **先拉这版重跑 depth=51/dnum=4/(4,4)。** 只差 0.8 GiB，而自适应 slab 正好省的就是这种整块浪费。
   这是成本最低、最可能直接通的一步。
2. 如果还差一点：**真正的解法是让空 slab 回到 driver**。需要记录每个 slab 的基址和它的块是否全空，
   `GPUfree` 时若整块空闲就 `cudaFreeAsync` 掉。这是个实打实的改动，我没 GPU 验不了，建议你那边评估。
3. **算法侧：已经做了，见下。**

---

## 六、追加：插一次自举，最小 depth 52 → 34，两堵墙相交了

先量清楚那 37 层花在哪（从 softmax 自己的自举往下数）：

| 段 | level | 累计 |
|---|---:|---:|
| softmax 尾部 | 14 | 14 |
| attention context | 2 | 16 |
| attention dense | 3 | **19** |
| LayerNorm | 14 | 33 |
| FF1 | 3 | 36 |
| GELU 自举前那次 rescale | 1 | 37 |

中间没有任何东西刷新**数据通路**——LayerNorm 内部那次自举刷的是**统计量**，不是值本身。
插一次就够，而让较长的一半最小的切点正好在 **stage 10 之后**：前 19、后 18。

`LayerNormStages.refresh` 就是它，`EncoderLayer(refresh_after_dense=True)` /
`bench --refresh-after-dense` 打开。它先把成对的实密文折成复密文，所以 8 条只花 **4 次自举**；
而且是**严格恒等变换**——自举前的 ×½ 和 `x + conj(x)` 的加倍互相抵消，和 stage 13 一模一样。

实测：

| | 最小 depth | N=2^16 / dnum 4 / (3,3) 下一层的显存 |
|---|---:|---:|
| THOR 原调度 | 52 | 35.3 GiB —— 装不下 |
| `--refresh-after-dense` | **34** | **27.4 GiB，余量 4.6 GiB** |

而且真实 MRPC 样本上跑出来的 logits 和不加它的那次**逐位相同**
（MAE 3.729e-05、概率 L1 9.077e-07），确认没有拿精度换深度。

代价：18 次自举变 22 次。默认关闭（这是相对 `he.py` 的偏离），但它是目前唯一能把一层塞进卡里的办法。

**所以下一次 GPU 运行建议直接用这组**：

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 --per-stage \n    --binary-rotations --refresh-after-dense --depth 34 --dnum 4 \n    --bootstrap-level-budget 3,3 --light-plaintext-cache 8
```

`depth=34` 下 `L+K` 大约 44，离 MAXP=64 很远；余量 4.6 GiB 也高于 keygen 需要的 3 GiB。
配上自适应 slab，这组应该是目前最有希望跑通的。

---

## 七、给 fidelity 报告的两条备注

* 「Stage 02 全部完成，无误差报告（rotate 是精确置换）」——精确的是**明文语义**；密文侧每次旋转
  是一次 key-switch，会加噪。binary rotations 下一次逻辑旋转变成最多 15 次，噪声按次数累积。
  stage 01–05 只用到 `-group_size` 这一种旋转（拆成 1 次），所以现在看不出来；等
  `rotate_internal` 和 `interval_sum` 跑起来（stage 06+），这一项值得单独量一下。
* depth=44 那组已经不用再试了：(3,3) 下 level 墙是 depth≥52。stage 01–05 能过是因为它们在第一次
  自举之前，用不到那 37 层。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CudaUtils.cu` | slab 分配失败逐级减半重试；检查 `cudaMallocAsync` 返回值；切分边界收紧；失败时报真实空闲量 **未编译** |
| `python/thorfhe/budget.py` | 加入 (4,4) 实测；密钥量改为锚定实测值；记录 `bootstrap_depth`；重写 "everything else" 的注释 |
| `python/tests/test_gpu_resource_fixes.py` | (4,4) 复现用例；未测量的例子换成 (5,5) |
