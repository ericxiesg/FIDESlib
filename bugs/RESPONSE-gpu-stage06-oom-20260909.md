# 回复 GPU-OOM-stage06-depth37-20260909

日期：2026-09-09。本机没有 GPU，**C++ 改动一行都没编译过**。

这一轮四件事同时被确认对了：**自举完整跑通、密钥计划正确（0 grown）、level 预算正确（depth=37
的自举后 level 20 够用）、stage 01–05 通过**。剩下的纯粹是显存，而且是我早就欠着的那笔账。

两边都改了：**分配器把空 slab 还给 driver**，以及 **stage 06 自己的峰值砍掉一大块**。

---

## 一、先排除一个选项

你的方案 4（depth=35/36）**不行**，这个我量过：一层需要的自举后 level 是 **20**，
19 就挂（`bootstrap_level=19 FAIL / 20 OK`）。depth=36 给 19，差一层。所以 depth 只能是 37 起步，
方向必须是省显存，不能是降 depth。

方案 3（少截断密钥）也是反的：不截断更费显存，而且你日志里 `0 grown at runtime` 说明现在这 16 把
截得**正好**——多一分会 grow，少一分白占。这条已经到最优了。

## 二、分配器：空 slab 现在会还给 driver

这就是我一直欠着的那条「真正的解法」。做法是**只在分配即将失败时**才去回收，所以快路径一行没动：

1. 记录每个 slab（基址、大小、块大小、块数）；
2. `GPUmalloc` 逐级减半仍然要不到时，先扫一遍所有 slab，把**所有块都在空闲表里**的整块还给 driver
   （`cudaFreeAsync`，因为它是 `cudaMallocAsync` 来的），然后**重试一次**；
3. 还是要不到才抛异常，错误信息也改了，不再说「从不归还」——现在会说「已经归还过，剩下的确实在用」。

回收时会打一行：

```
[FIDESlib] memory pool returned 512 MiB of wholly free slabs to the driver
```

你这次的场景正是它针对的：**512 KiB 那个 size class 一个空闲块都没有、也放不下新 slab，
而 16 MiB 全躺在别的 class 里**。

## 三、stage 06 自己的峰值：一次少 127 条密文

顺着堆栈看了 `_accumulate_product`，发现一处比分配器更值钱的：

```python
diagonals = [level_down(ct, ...) if ... else ct for ct in diagonals]
```

`diagonals` 在 THOR 几何下是 **128 条密文**。这一行把它们**全部**降级成新对象，
而原来那 128 条还被调用方引用着——于是同时活着 256 条。depth=37 下一条密文是
2 × 38 limb × 512 KiB ≈ **38 MiB**，多出来的 128 条就是 **约 5 GiB**。

而每条 diagonal 在下面的循环里**只用在一次迭代中**。所以改成在那次迭代里就地对齐、用完即弃：
**同时活着的从 128 条降到 1 条**。

`left` 只有 2–4 条，仍然一次性对齐。数值完全不变（对齐本来就是逐条独立的），
attention 那 13 个用例照过，另加了一个用例钉住「不会在第一次乘法之前把 128 条全降完」。

这一条不需要编译，是 Python 侧的，**下次跑立刻生效**。

## 四、还能再省的（按性价比，都还没做）

1. `--light-plaintext-cache 4`：depth=37 下每项约 20 MiB，8→4 省 80 MiB。你崩在 16 MiB 上，
   这 80 MiB 未必不够用。**下次可以顺手带上。**
2. 辅助 poly 池（`trimAuxilarPoly` 至今没人调用）：以前调了也没用——归还只是回到 size class 空闲表。
   **但现在不一样了**：有了 slab 回收，排空 poly 池就可能真的腾出整块 slab。这两个修改是**相乘**的。
   要做的话是加一个 API + 在 stage 边界调用，我可以下一轮做。
3. `levelBudget={4,4}`：明文从 7.6 GiB 降到约 6.9 GiB，但 bootstrap depth 从 17 涨到 18，
   depth 要 38——净收益要实测才知道。

---

## 五、下一次运行

先跑原来那组，**只加一个 cache 参数**：

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 --per-stage \
    --binary-rotations --refresh-after-dense --depth 37 --dnum 4 \
    --bootstrap-level-budget 3,3 --light-plaintext-cache 4
```

三件事请注意看：

* 有没有出现 `memory pool returned ... MiB` 这一行——有的话说明回收起作用了；
* stage 06 过没过（Python 侧那 5 GiB 的削减本身可能就够了）；
* 跑通的话，**这是第一次在 GPU 上跑完 THOR 一整层**，请发回 `--per-stage` 那张表、单层耗时、
  峰值显存、`key memory` 那一行。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CudaUtils.cu` | 记录 slab；分配失败时把整块空闲的 slab 还给 driver 再重试一次；只在失败路径上跑 **未编译** |
| `python/thorfhe/attention.py` | `_accumulate_product` 逐条对齐 diagonal，峰值从 128 条降到 1 条 |
| `python/tests/test_gpu_resource_fixes.py` | 新增用例钉住惰性对齐 |
