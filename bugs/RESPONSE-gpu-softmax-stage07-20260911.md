# 回复：逐级表钉死了 stage 07，再给两个探针把它切成四段

日期：2026-09-11。**C++ 本轮没改**（本机无 GPU，上一轮那三处还没编译过）。

## 一、先认错两条

**（1）我那条「`EvalNegate` 的 scale bug 就是保真度问题」是错的。** 你用 v6 与 v8 结果完全相同
证伪了它，证得干净。`he_invsqrt` 在 LayerNorm 里，在 softmax 下游，softmax 已经坏了它就没机会表态。
那个 scale bug 是真的、该修，但**不是这个病**。

**（2）不过 `addPt/subPt` 守卫没报错是有信息量的负结果**：整层跑下来**没有任何一次明文加法
scale 不匹配**。这一次性排掉了一整类嫌疑，别再往那边找了。

## 二、逐级表其实已经把范围缩得比报告里更小

stage 06 的 `scores` 是 **relRMSE 2.873e-07**，完美。从它到 `07a` 之间，代码只有这些：

```python
merged    = add(scores[i], multiply_1j(scores[i + half]))
merged    = bootstrap(merged)                    # <-- 只有它是新的
merged    = level_down(merged, 3)
conjugated= conjugate(merged)
refreshed[i]        = add(merged, conjugated)
refreshed[i + half] = multiply_1j(subtract(conjugated, merged))
```

`multiply_1j`、`conjugate`、`add`、`subtract`、`level_down` **在 stage 06 内部全都用过**，
而 stage 06 的输出是 2.9e-7。所以：

> **通往 07a 的路径上，唯一一个前六级没有验证过的算子就是 `bootstrap`。**

你列的 5 个候选里，这一条把 #1 抬到了第一位，也顺便把 #5（doubled scores 的处理）排到后面——
score 的加倍发生在 `add(merged, conjugated)`，而那两个算子是验证过的。

## 三、两个探针，一次跑把 stage 07 切成四段

新增 `Stages.probe`（默认 `None`，普通跑不触发），`--per-stage` 时自动挂上。
它**只报量级和 level，不需要任何参照**——stage 内部的中间量在明文模型里没有对应物，
但「本该是概率却回来 1e12」「本该随 token 变化却是常数」「level 和排程对不上」这些，
不用参照也能看出来。

四个探点：`07a` bootstrap 之后、`07b` `he_exp` 之后、`07c` 分母、`07d` 分母的倒数。

**本机 clear engine 的基准剖面**（精确算术，同样的权重和样本）：

```
[probe] 07a.refreshed_scores      8 ct  min -10.41  max +12.43   used 196608  |x| med 1.522
[probe] 07b.exp                   8 ct  min +0      max +0.000228 used  67584  |x| med 1.35e-06
[probe] 07c.denominator           1 ct  min +0      max +0.0005861 used 24576  |x| med 8.061e-05
[probe] 07d.inverse_denominator   1 ct  min +0      max +36.23   used 24576  |x| med 30
```

（level 数字不用对比——clear engine 的 depth 和 GPU 的 37 不是一回事；**要比的是量级**，
以及各探点之间 level 下降了多少。）

**怎么读：**

* `07a` 回来还是 `[-10, +12]`、`|x| med` 1.5 左右 → **bootstrap 是好的**，问题在它之后；
* `07a` 就已经不成样子 → **就是 bootstrap**，和第二节的推理一致，直接去查它；
* `07a` 正常但 `07b` 不对 → `he_exp`（见下一节，这里有个独立的疑点）；
* `07c` 正常但 `07d` 不对 → Goldschmidt 除法不收敛。

## 四、顺便：`07b` 这一行本身就可疑，不管 bootstrap 结论如何

注意 clear engine 上 `he_exp` 的输出是 **max 2.28e-4、中位数 1.35e-6**。这是**刻意做小的**——
`he_inv` 要求分母落在 `[epsilon, 1]`，所以整条链在很小的量级上跑。

但在 CKKS 里这有代价。你们自己测过的 **GPU bootstrap 绝对误差是 1.74e-5**。
如果这个参数集的噪声底噪也在 1e-6 ~ 1e-5 这个量级，那么：

* `07b` 最大的那些值（2.28e-4）只剩一两位有效数字；
* **中位数那一批（1.35e-6）整个在噪声底噪以下**。

分子分母都在噪声里，它们的比值就是**完全不相关**——这正好是 `scale 0.0000 / relRMSE 1.0` 的样子，
而不是「精度差一点」的样子。

所以 `07b` 的 `|x| med` 和 `max` 请一并发回来。**如果 GPU 上 `07a` 正常而 `07b` 的量级对、
数值却不对，那就是噪声底噪问题，不是逻辑 bug**，解法是另一套（抬 scaling_bits，
或者把 `he_exp` 的输出整体乘一个整数放大、在 `he_inv` 前再缩回来——整数乘不耗 level）。
先别急着改，等数据。

## 五、下一跑

命令不变，探针跟着 `--per-stage` 自动开：

```bash
python3 -u -m thorfhe.bench fhe --engine fideslib --device cuda:0 \
    --depth 37 --dnum 4 --bootstrap-level-budget 3,3 \
    --binary-rotations --refresh-after-dense \
    --layers 1 --limit 1 --per-stage --device-memory --offline
```

把四行 `[probe]` 发回来就够了，逐级表是附赠的。

顺带：`intermediate` 之后那个 `SetLevel: multiplicative depth [37] is insufficient` 是
**解码**时才炸的（`[mem]` 显示 15 个 stage 边界全过了），所以它是 level 记账的问题，
不是计算没跑。等 softmax 定位完再看它——在 softmax 修好之前，那之后的 level 本来就是乱的。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/stages.py` | `Stages.probe` / `Stages.probed()`，默认关闭 |
| `python/thorfhe/softmax.py` | stage 07 内部四个探点 |
| `python/thorfhe/bench.py` | `--per-stage` 时挂上量级探针（量级 + level，不需要参照） |
