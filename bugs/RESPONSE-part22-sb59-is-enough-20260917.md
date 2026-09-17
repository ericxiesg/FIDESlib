# Part22 核对：诊断对，但 §4 的第 1 条是空操作，§6 的悲观是算错了

日期：2026-09-17。针对 `89b5479` 和 Part22。

---

## 0. 结论

| Part22 的说法 | 核对结果 |
|---|---|
| MAE 7.5e17 是 bootstrap 绝对误差把分母炸了 | **对**，而且我复现了 |
| `noise_model` 没接进 bench | **对**，已接（`--noise-model`） |
| §4.1 传 `numIterations=2, precision=10`，"改一行 binding" | **空操作**，GPU 路径把这两个参数丢掉 |
| §6 "Δ 50→59 不够" | **算错了**，59 下实测误差是 1.55e-6 不是 11 bit |

**一句话：不需要移植 CPU 的 Meta-BTS，需要的是 sb=59——而那本来就是计划。
实测在 sb=59 下，设备自己的 bootstrap 误差在 softmax 输出上看不见。**

---

## 1. `numIterations` 在 GPU 路径上被丢掉了

`api/CryptoContext.cpp:1754`：

```cpp
Ciphertext<DCRTPoly> CryptoContextImpl<DCRTPoly>::EvalBootstrap(
        const Ciphertext<DCRTPoly>& ciphertext,
        uint32_t numIterations, uint32_t precision, bool prescaled, int stopAfterStage) {
    if (this->devices.empty()) {                       // ← CPU fallback，这里才转发
        auto ct = context->EvalBootstrap(ctImpl, numIterations, precision);
        ...
    }
    // GPU path
    FIDESlib::CKKS::Bootstrap(*res_gpu, res_gpu->slots, prescaled, stopAfterStage);
    //                        ^ 没有 numIterations，没有 precision
}
```

`FIDESlib::CKKS::Bootstrap` 的签名（`Bootstrap.cuh:21`）里根本没有这两个参数。

**所以传 `numIterations=2` 会静悄悄地什么都不做**——跑一遍，结果一模一样，
然后花一天找原因。Meta-BTS 在 FIDESlib 里是**要实现**，不是要打开。
（可以在 Python 层用现有原语搭：一次 bootstrap、减出残差、整数放大、再 bootstrap、缩回相加。
代价是每个站点多一次 bootstrap 和大约一层。**但下面第 3 节说明不需要。**）

---

## 2. §6 的"不够"是把 11 bit 当成常数了

**设备的 bootstrap 精度不是一个常数，它随 `q0/Δ` 变。** 这是上一轮就量过的
（`RESPONSE-migration-two-gaps-20260916.md` §0：你那边实测 `C = 2^39.7, error = 1.55e-6, 20.3 bit`）：

```
  参数           q0/Δ    绝对误差    相对界的 bit 数
  sb=50 fmb=55     32   1.56e-02      11.0     <- Part22 量到的地板就是这里
  sb=55 fmb=60     32   4.88e-04      16.0
  sb=59 fmb=60      2   1.55e-06      20.3     <- 上一轮你自己实测的
```

Part22 §0 那张表里"59/60 + bootstrap 11 bit"这一行，**设备不会那样**——
11 bit 是 `q0/Δ=32` 下的地板，59/60 的 `q0/Δ` 是 2。

而且**bit 数不能跨参数集比较**，因为每个都是相对各自的界。换成绝对误差，
Part22 自己那张悬崖表是这样的：

```
  sb=50, 22 bit  ->  绝对 7.63e-06   活，max|err| 7.6e-3
  sb=50, 18 bit  ->  绝对 1.22e-04   勉强
  sb=50, 16 bit  ->  绝对 4.88e-04   死
  sb=50, 11 bit  ->  绝对 1.56e-02   死透

  sb=59, 实测    ->  绝对 1.55e-06   比"活"那一行还好 5 倍
```

---

## 3. 实测：sb=59 下噪声在 softmax 输出上看不见

把 `noise_model` 接进 `make_engine` 之后（`--noise-model`、
`--bootstrap-precision-bits`，默认 11 = 设备实测地板），真实 checkpoint 的 score、
真实 attention mask、`Softmax.LAYERS` 的逐层参数：

```
layer 0, MRPC row 0, 44 real tokens
  参数              bit   绝对误差    max|softmax - 真值|
  exact (无噪声)      0   0.00e+00           0.007091
  sb=50 fmb=55       11   1.56e-02         1.782e+184     <- 复现了 GPU 那次
  sb=55 fmb=60       16   4.88e-04            0.01967
  sb=59 fmb=60       20   1.91e-06            0.00709     <- 和精确算术三位有效数字相同

layer 8（全网最难的一层）
  exact               0   0.00e+00           0.002198
  sb=50              11   1.56e-02                nan
  sb=55              16   4.88e-04           4.44e+55     <- sb=55 在这层是死的
  sb=59              20   1.91e-06            0.03415     <- 活，但比精确差 15 倍
```

**sb=59 够，sb=55 不够**（layer 8 上 sb=55 直接炸）。这条线索也说明为什么
只看 layer 0 会得出"sb=55 也行"的错误结论。

---

## 4. layer 8 改走 wide 多项式（已改）

layer 8 在 sb=59 下还差 15 倍，根因是它的分母跨度 min/max = 1.2e-4，全网最宽，
而且**和窗口中心无关**。`he_exp2` 的斜率是一半，所以同样的 score 过去，
跨度变成原来的**平方根**：

```
layer 8，sb=59，对真 softmax：
  narrow（原来）   0.0022 精确  ->  0.0342 带噪声     噪声占主导
  wide,  l=4      0.0070 精确  ->  0.0070 带噪声     噪声看不见
```

**wide 在精确算术下差 3 倍，在设备上好 5 倍。** 设备上那个才算数。
顺带 Goldschmidt 从 10 次降到 6 次，12 层总迭代 86 → 82。

`LAYERS[8]` 已改成 `dict(WIDE, shift=-23.25, inv_epsilon=2**-8)`。
（layer 2 是被迫 wide——narrow 在它的 score 上任何中心都溢出；layer 8 是**选择** wide。）

---

## 5. 所以方向

1. **sb=59 / fmb=60，不要折中。** sb=52/55 在 layer 8 上是死的（899%/1272% 的误差对分母）。
   你 §5 问的"sb=52 折中"——不行。
2. **不要传 `numIterations=2`**，它不会做任何事。
3. **性能才是唯一的阻塞**。这一条我在 `RESPONSE-denominator-is-the-softmax-scale` 里说过：
   注意你那次"快"的 sb=50 跑，GPU util 也只有 1-3%——**这个 workload 在能跑通的时候
   也是 host-bound 的**。0-4% 不是旋转次数的症状（旋转多会表现为 util 高、时间长）。
   25.2 GB 有 6.8 GB 余量，29.3 GB 只有 2.7 GB——**在 91% 占用下每次分配都要走 driver，
   而 cudaMalloc/cudaFree 是同步的**，表现就是 0% util + 一个 CPU 核满载。
   请带 `--device-memory` 跑一次 sb=59，把每个 stage 的 `pooled / in_use / driver_free` 报出来。
4. **`--extra-rotation-keys 6` 先别加**：它给一个已经在 91% 的跑再加 1 GB。
5. **还有一个没解释的**：sb=50 是 25.2 GB，sb=59 是 29.3 GB。`budget.py:key_bytes`
   没有位宽参数——一个 RNS tower 不管素数是 50 位还是 59 位都是每系数一个 64 位字。
   这 4 GB 的差是哪来的？如果两次跑的 `--depth` 或 level budget 不一样，说一声。

---

## 6. 我这边还改了

* `--noise-model` / `--bootstrap-precision-bits` 接进 `make_engine`（Part22 §4.2，确实是个洞）。
* `LAYERS[8]` 改 wide。
* Part22 §1.2 说的比对口径（`decode_six_blocks` 不按 mask 切）**还没改**——
  你说得对，两边 MAE 不是同一个量。但它不是这次的病因，我先不动，
  免得和精度的改动混在一起。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
