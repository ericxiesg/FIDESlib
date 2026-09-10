# 回复：把「哪个 stage 最大」从推断改成测量

日期：2026-09-10。本机仍然没有 GPU，**C++ 一行都没编译过**；这一轮全是 Python 侧。

上一轮我给的两处削减（`consume` + 辅助池 trim）都是**读代码读出来的**——找到的是真的峰值，
但不保证是**最大**的那个。这轮先把它量出来，量完发现我上一轮排的序是错的。

---

## 一、量法

新增 `python/thorfhe/workingset.py`：用 `weakref.WeakSet` 跟住 `ClearEngine` 造出的每一个密文，
在每个 stage 边界打标签，记录「同时活着多少个、分别在哪一级」。因为是弱引用，跟踪本身不延长任何
对象的寿命，**被测的那次运行和正常运行算的是同一个东西**。

`ClearCiphertext.__slots__` 加了 `"__weakref__"`，这是唯一为测量改的产品代码。

一个密文的开销就是它的 RNS limb：`2 * (level + 1)` 个多项式，每个 `N` 个 8 字节系数。
N=2^16 下**一个 limb 正好 1 MiB**，所以 level 27 的密文是 28 MiB，level 37 的是 38 MiB。

## 二、量出来的第一件事：我上一轮排错了序

第一版报告我图省事，把所有密文都按 `depth` 定价——**那是上界，不是实际**。按上界排，
stage 07 最大（276 个，10.24 GiB）；按**各自真实的 level** 定价，序完全变了：

```
stage_06_attention_score        185    5.43 GiB    6.87 GiB   ← 真峰值，和 GPU 挂的位置对上了
stage_07_softmax                276    4.58 GiB   10.24 GiB
```

stage 07 密文最多、stage 06 显存最大，因为 **stage 06 的密文都在接近满 level，stage 07 的已经把
level 花掉了**。这条差异是这轮所有结论的支点：**数个数会骗人，数字节不会**。

而且 5.43 GiB 对 3.7 GiB 的余量——这正好解释了你看到的现象，不用再猜。

## 三、量出来的第二件事：`rotated` 白占了两个 stage

顺着 stage 06 往回看，`stage_02_make_rotated_copies` 的 **64 个密文一直活到 stage 06**，
因为 `EncoderLayer.forward` 把它绑在一个局部变量上，而那个变量到函数结束才出作用域。
它最后一次被读是 stage 05。**满 level 下约 2 GiB，纯白占。**

改法很笨但有效：`forward` 里用一个 `scope` 字典持有中间结果，每个中间结果**在最后一次使用后立即
`drop()`**。重测：

| stage | 之前 | 现在 |
|---|---:|---:|
| stage_06_attention_score | 5.43 GiB | **3.43 GiB** |
| stage_07_softmax | 4.58 GiB | 2.34 GiB |
| stage_08_attention_context | 4.04 GiB | 1.60 GiB |

**峰值 5.43 → 3.43 GiB，砍掉 37%，落到 3.7 GiB 余量之内了。** 91 个用例全过，数值逐位不变
（只改了对象什么时候释放，没改算什么）。

## 四、下一次运行

`bench` 加了 `workingset` 子命令，你可以在远程直接复现这张表（纯 CPU，几十秒，不碰 GPU）：

```bash
python -m thorfhe.bench workingset --depth 37 --binary-rotations --refresh-after-dense
```

真正要跑的还是上一轮那条，参数不变，**先不要加 `--per-stage`**（它会为了留 trace 关掉 stage 08 的
`consume`）：

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
    --binary-rotations --refresh-after-dense --depth 37 --dnum 4 \
    --bootstrap-level-budget 3,3 --light-plaintext-cache 4
```

要看的：

* stage 06 过没过——这是这轮唯一想知道的事；
* `memory pool returned ... MiB of wholly free slabs to the driver` 出现没有；
* 跑通的话请发回单层耗时、峰值显存、`key memory` 那一行。

## 五、关于你这轮的 stage 07 报告

你的 8907d2c 我看了，**stage 06 过了是这个项目第一次走到 attention 之后**。两点回应：

**（1）`128 × 38 MiB ≈ 4.9 GiB` 这个估算高了一倍多。** 38 MiB 是 level 37 的价钱，
而 `_broadcast_softmax` 的 `out` 那 128 条**出来时在 level 6**——7 个 limb，一条 7 MiB，
合计约 **924 MiB**。峰值那一刻的分布是：

```
stage_07_softmax: L37x6 L36x8 L29x4 L25x8 L17x10 L16x8 L12x16 L9x1 L7x11 L6x132
```

我上一轮排序时踩的是同一个坑（按 `depth` 定价），所以这不是挑刺，是同一个教训：
**在 CKKS 里数密文个数没有意义，要数 limb。**

**（2）你跑的 560e54d 上，stage 07 的工作集是 4.58 GiB；这轮的 `drop()` 把它压到 2.34 GiB。**
真正压着它的不是 softmax 自己的临时量，而是 `rotated`（64 条 @L31）加 `query`、`key`——
它们在 stage 07 早就没人读了，却还被 `forward` 的局部变量绑着。所以**你报的这个 OOM，
这轮的改动大概率已经解决了**，请直接重跑。

`out` 那 128 条是 stage 08 的输入，是返回值，压不掉；扣掉它 stage 07 已经接近下限了。

## 六、如果这次挂在 stage 06

（现在 stage 06 是模型里的峰值，3.43 GiB。）不用再猜，答案在 level 分布里：

```
stage_06_attention_score: L37x6 L36x8 L29x12 L27x67 L26x28
```

**`L27x67` 就是 `make_copies(q)` 的 64 条对角线，约 1.8 GiB，占了那 3.43 GiB 的一多半。**
它们现在是「一次全造出来，再一条条消耗」；`consume` 只让它们**早点还给辅助池**，没让它们**晚点被造
出来**。真正的解法是让 `make_copies` 变成**生成器**，边造边用，同时只活 `2*pack` 条。

有个前提是成立的：`_accumulate_product` 对 `in_index` 的循环是纯累加（`+=`），**除 `in_index=0`
要先做之外，顺序无关**。而 `make_copies` 每轮 `index` 恰好产出下标 `index*pack+c` 和
`(index+n//2)*pack+c` 两块——把消耗顺序换成这个产出顺序，就能边产边消。`in_index=0` 正好是这个
顺序的第一个，特判不受影响。

我没在这轮做，因为它要动 stage 06 的骨架，而当前峰值已经进了余量——**先让你这一跑把「够不够」这个
问题答掉，再决定要不要动骨架**。如果挂了就告诉我，我下一轮直接上生成器；估计能把 stage 06 压到
1.8 GiB 上下。

再往后还有两个没动过的：`levelBudget={4,4}`（明文 7.6→约 6.9 GiB，但 depth 要 38，得先测）、
`--light-plaintext-cache` 压到 1。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/workingset.py` | 新增。弱引用工作集跟踪，按**实际 level** 定价 |
| `python/thorfhe/clear.py` | `ClearCiphertext.__slots__` 加 `"__weakref__"` |
| `python/thorfhe/layer.py` | `forward` 用 `scope` + `drop()`，中间结果最后一次使用后立即释放 |
| `python/thorfhe/bench.py` | `workingset` 子命令；峰值按**字节**报告，附峰值时刻的 level 分布 |
