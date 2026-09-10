# 追加：stage 06 的 64 条对角线不再一次性造出来

日期：2026-09-10，第三份，接在 [device-memory 探针](RESPONSE-gpu-device-memory-probe-20260910.md) 后面。
**纯 Python，没有新的 C++ 改动**（前两份里的 C++ 仍然未编译）。

上一份说「如果还挂在 stage 06 我下一轮上生成器」。既然探针那一版还要等你跑，我就先把它做了——
反正 stage 06 是连挂两轮的地方，早一轮拿掉早安心。

## 做了什么

`make_copies` 拆成 `iter_copies`：**边造边交**，产出 `(position, ciphertext)` 而不是填满一个数组。
`make_copies` 本身留着（测试和别处还在用），实现改成把生成器灌进数组，行为完全不变。

能这么做的前提是 `_accumulate_product` 的循环**只做累加**，所以对角线到达的顺序无所谓。
原来 `in_index == 0` 是循环外的特判（直接赋值给 `accumulator[i, 0]`），我把它改成走 `accumulate()`——
两条路径产出的项本来就在同一 level（前者 `multiply` 后 `level_down(by=1)`，后者 `rescale` 后乘掩码），
所以这只是**去掉一个对顺序的隐含依赖**，不是新增逻辑。改完整个循环真的与顺序无关，生成器才敢接。

要留意的一点：`iter_copies` 每轮产出的是**下标 `index` 和 `index + n//2` 两块**，因为一个复数密文的
实部虚部是两条不同的对角线。所以产出顺序不是自然顺序，交出去的必须是「下标 + 密文」而不是一串。

`_accumulate_streamed` 唯一做不到的是**开始前扫一遍所有对角线取公共 level**，所以它取第一条的 level
并**检查**后面每一条都一样。这里所有生产者都是一条统一路径造出来的，本来就成立；写成检查而不是假设，
是因为万一不成立，它会以「FIXEDMANUAL scale 不匹配」的形式在很靠后的地方才炸出来。

## 效果

```
stage_06_attention_score: L37x6 L36x8 L29x13 L28x5 L27x10 L26x33     3.43 → 2.18 GiB
```

那 64 条 `L27` 现在同时只活一条。**stage 06 降了 36%，它不再是这一层的峰值了。**
91 个用例全过——这些用例是逐 slot 对公式的，所以输出逐位相同。

## 现在的峰值换人了

```
peak 3.04 GiB in stage_05_value (96 live)
stage_05_value: L37x6 L36x8 L31x64 L29x18
```

`L31x64` 是 stage 02 的 `rotated`，**2 GiB，从 stage 02 一直活到 stage 05**——因为 stage 03/04/05
（Q、K、V 三个投影）都要读它。这个我**没有**动，因为拿掉它要把三个 stage 融成对 `rotated` 的一遍扫描，
是改骨架，而且会动到逐 stage 对拍——那是这个项目唯一的正确性工具。

**先等你的 `--device-memory` 数据。** 如果数据说真正占满卡的是池子在囤而不是工作集，那融合 stage
03/04/05 就是白改。

## 下一跑

命令和上一份完全一样（记得带 `--device-memory`）。这轮之后主机侧模型里：

| stage | 工作集 |
|---|---:|
| stage_05_value | 3.04 GiB |
| stage_04_key | 2.92 GiB |
| stage_03_query | 2.80 GiB |
| stage_02_make_rotated_copies | 2.51 GiB |
| stage_07_softmax | 2.34 GiB |
| stage_06_attention_score | 2.18 GiB |

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/attention.py` | `iter_copies` 生成器；`_accumulate` 与顺序无关；`_accumulate_streamed`；stage 06 改用它 |
