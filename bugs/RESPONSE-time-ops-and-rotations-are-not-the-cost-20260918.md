# 旋转不是瓶颈（你的数据证明的），请跑 `--time-ops`

日期：2026-09-18。针对 `1413375`。

---

## 0. 你这次的两行数据把问题定死了

| 基 | 旋转/层 | wall clock |
|---|---|---|
| binary，15 把 | 8258 | 1161.8 s |
| +6，max_steps=4 | 3643 | 1154.7 s |

少 4615 次 key-switch，省 **7.1 秒**。

- 每次旋转 **1.54 ms**
- 全层 3643 次旋转 = **5.6 s = 0.5%**

**旋转做到 0 也只省 0.5%。** 我之前那份 perf 分析里"批量旋转预计加速 2–4x"的判断
被你的数据否定了，撤回。别再在旋转上花时间。

---

## 1. 剩下 1149 秒只有两种可能，`--time-ops` 一次跑分得开

一层 **80,065** 次 engine 调用（本机 ClearEngine 实测，调用序列和设备逐条一致）：

```
multiply 23314   add 22660   add_inplace 20486   level 6160   rescale 2760
rotate 1922      level_down 1336   relinearize 659   其余 726   bootstrap 22
```

| 假设 | 每次代价 |
|---|---|
| 22 次 bootstrap 吃掉 1149 s | **52 s / 次** |
| 78,143 次逐元素调用吃掉 1149 s | **14.7 ms / 次** |

两个都比合理值高两个数量级。**这两条路的优化方向完全相反**，所以在分开之前不要动任何东西。

`a22ec04` 之后我加了 `--time-ops`（已推 `…`）：包住每个 engine primitive，
计数 + 计时，每次约 0.2 µs（对 14 ms 的调用是 1/70000），最后打一张表：

```
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 --time-ops \
  --refresh-after-dense --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --extra-rotation-keys 6

engine primitives (1 encrypted layers, 1 samples)
  primitive             calls    seconds   share    us/call
  ...
```

**请跑这一条，把表贴回来。** 另外代码里已经有 `CudaNvtxRange`，
`nsys profile -t cuda,nvtx` 跑一层能直接给内核级归属，如果机器上有 nsys 就一起跑。

---

## 2. 回答你的三个问题

### Q1：variance 的修法方向

**先别改算法。** 我在 `44c5312` 里把相减前的两项分别探出来了（你那份报告发在这之前）：

```
11a1.n_sum_of_squares   max 0.2161
11a2.squared_total      max 0.001773     <- 是前者的 1/122
11a.variance            max 0.2161
```

`(Σx)²` 只有 `n·Σx²` 的 1/122，相减拿掉 0.8%，**条件数 1.008**。
`values` 掩码已经先除掉了 `sqrt((1.05·max_var+var_e)·n²)`，LayerNorm 输入的均值又接近 0，
所以 `(Σx)²` 项一开始就小。你说的"两个都是 ~1e9 量级"在 clear 端不成立——**都是 0.2 量级**。

更硬的一条：**相消只会让结果变小。** 两个被减数最大 0.216，你量到 4.6e7，
比它们都大 2.1 亿倍。这个数不可能是这一行减出来的，**是它的某个输入在进这一行之前就错了 8 个数量级**。

`a22ec04` 起 `--per-stage` 会打这四行，请按顺序看第一个对不上的：

```
10.attention_dense      clear: level 1   max +5.23
11.residual             clear: level 20  max +10.76   (min -26.72)
11a1.n_sum_of_squares   clear: level 17  max +0.2161
11a2.squared_total      clear: level 17  max +0.001773
11a.variance            clear: level 17  max +0.2161
```

- `11.residual` 就炸 → 问题在 stage 10 / residual，和 LayerNorm 无关
- `11.residual` 正常、`11a1` 炸 → 在 `masked` / `square` / `_fold_into_slot_zero` / `statistic`
- 两项都正常、`variance` 才炸 → 那时才轮到 `subtract`

**A 方案（改 `Σ(x−mean)²`）现在没有理由做**：条件数 1.008 买不到精度，还要多一轮 broadcast 和一层 level。
**B 方案（Meta-BTS）先定价再决定**：+4 bootstrap/层，如果 bootstrap 真是 52 s/次，那是 **+208 s/层**。
这也是为什么 §1 那张表是精度方案的前提。

### Q2：21 把密钥下 `07d` 的 outlier

**这个要管，而且它比 variance 更像一条新线索。**

```
15 把：07d.inverse_denominator  max 0.1477            干净
21 把：07d.inverse_denominator  min -3099  max 6.674e16  p50 0.02425  p99 0.2426
```

p99 正常、max 炸，是**少数 slot 拿到了错的值**，不是噪声——噪声会抬高整条分布。
而且方向反了：21 把密钥的链**比** binary **短**（3643 vs 8258 次 key-switch），
噪声应该更小才对。

所以嫌疑是**某个 index 的分解落到了错的 slot 或错的 level**。请帮我缩小：

1. 用 `--extra-rotation-keys 6 --rotation-max-steps 2` 跑同一条（4256 次旋转，同样 21 把密钥）。
   - **还炸** → 是 21 把这套密钥本身的问题（key level plan），和 meet-in-the-middle 无关
   - **不炸** → 是 3 步以上的链，我这边去查 `_meet`
2. 加 `--no-truncate-keys` 再跑一次 21 把 + max_steps=4。
   - **不炸** → 是 `key_levels` 给某把密钥定的 level 偏低
3. 带 `--allow-key-grow` 跑，看有没有打印"key grew"。不带它时超界应该抛异常，
   你没抛，所以要么 level 是够的、要么这条检查没生效——这一条能分开。

本机上 clear engine 一层的端到端是逐位相同的（`relRMSE 2.776e-03`，和一 index 一把密钥一致），
所以分解的**算术**是对的；能出问题的只剩密钥的 level 和设备端的 key-switch。

### Q3：要不要跑 `--rotation-key-budget 3.4`

**不用。** 29.2 GiB 已经跑通，budget 是"卡装不下时用 0.09 GiB 换 178 次旋转"的工具，
而 §0 已经说明旋转只值 1.54 ms。现在没有需要它解决的问题。

---

## 3. 另外两条，顺手的

- `src/CudaUtils.cu:301-302`：`cudaEventCreateWithFlags(&ev,…)` 之后紧跟
  `cudaEventCreate(&ev,…)`，**第一个 event 句柄被覆盖且永不销毁**。stream 有池（37 个），
  event 没有，每个 `LimbPartition` 构造一次。删掉其中一行，代价为零。
- `Ciphertext.cpp:570` 那段 `PRINT=true` + `cudaDeviceSynchronize` + 打印 limb 的块是死代码
  （外层 `if (0 && …)` 和 `if constexpr (0)`），**不是**热点，我查过了。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
