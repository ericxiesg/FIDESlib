# --time-ops 结果：encode_to_light_plaintext 占 96.2%

日期：2026-09-18。承接 `cb4a22b`。

---

## 0. 结论

**瓶颈是 `encode_to_light_plaintext`，不是 bootstrap、不是旋转、不是乘法。**

```
encode_to_light_plaintext    19,137 calls    1109.65 s    96.2%    58 ms/call
decrypt                         258 calls      19.74 s     1.7%    76 ms/call
multiply                     23,117 calls      10.73 s     0.9%   0.5 ms/call
bootstrap                        22 calls       4.16 s     0.4%   189 ms/call
rotate                        3,643 calls       0.77 s     0.1%   0.2 ms/call
add_inplace                  20,486 calls       0.82 s     0.1%   0.04 ms/call
total                        80,788 calls    1154.00 s
```

**去掉 encode 后一层只需 ~44 秒**（1154 - 1110 = 44s）。

---

## 1. 这解释了 GPU util 0-5% 的现象

`encode_to_light_plaintext` 是 CPU 操作：把 numpy array 编码成 light plaintext（多项式表示）
用于和密文相乘。19,137 次/层，每次 58 ms，全是 CPU 时间。GPU 在等。

之前报告（`RESPONSE-gpu-and-perf-analysis-20260917.md`）里说的"CPU dispatch 主导"
方向对了，但定位错了——以为是 pybind11 GIL 和 kernel 太小，实际上是 **plaintext encoding
本身就慢**，和 GIL 无关。

---

## 2. 两条假设被数据否定

协作者在 `cb4a22b` 里提了两条假设：

| 假设 | 预期 | 实测 |
|---|---|---|
| 22 次 bootstrap 吃掉 1149 s | 52 s/次 | **189 ms/次，总共 4.16 s = 0.4%** |
| 78,143 次逐元素调用吃掉 1149 s | 14.7 ms/次 | **multiply 0.5 ms, add 0.7 ms, rotate 0.2 ms——全都不对** |

**第三条没人想到**：19,137 次 `encode_to_light_plaintext`，58 ms/次 = 1109 s = 96.2%。

---

## 3. 为什么 19,137 次？

一层有 ~378 个权重矩阵（从 `Plaintexts loaded: 378 ~ 7749MB` 可见）。每个矩阵有 ~50 个
slot rotation 的 light plaintext（一个矩阵乘法需要把权重按不同 rotation 编码成多个
light plaintext）。378 × 50 ≈ 19,000，和 19,137 吻合。

`--compact` 模式本应"惰性权重编码"——但惰性编码发生在**计算过程中**，不是预编码。
每次需要权重时才编码，58 ms 一次，19,137 次就是 1110 秒。

---

## 4. 修法方向

### A. 预编码所有权重（推荐）
在 `run_encrypted()` 开始前，把所有 378 个权重矩阵一次性编码成 light plaintext，
存到 GPU 上。计算时直接取用，不需要实时编码。

- 预期：一层从 1154 s → ~44 s（**26x 加速**）
- 代价：更多 GPU 内存（编码后的 light plaintext 占空间），但 7749 MB 已经在 GPU 上了
- 风险：预编码本身也要 1110 s，但只需要做一次（可以在 key generation 阶段并行做）

### B. 把 encode 搬到 GPU
`encode_to_light_plaintext` 是 NTT/INTTT 操作，GPU 上可以并行做。但需要改 FIDESlib C++ 代码。

### C. 缓存 encode 结果
如果同一个权重被多次使用（比如多个 sample），缓存编码结果。但 THOR 一层每个权重只用一次。

**推荐 A**：预编码，不改 C++ 代码，只改 Python 层的调度。

---

## 5. 完整 --time-ops 表

```
engine primitives (1 encrypted layers, 1 samples)
  primitive             calls    seconds   share    us/call
  encode_to_light_plaintext    19137    1109.65   96.2%    57984.4
  decrypt                 258      19.74    1.7%    76494.8
  multiply              23117      10.73    0.9%      464.2
  bootstrap                22       4.16    0.4%   189219.1
  multiply_1j             174       2.20    0.2%    12657.8
  encrypt                   6       1.84    0.2%   305953.6
  encode                   79       1.70    0.1%    21506.8
  add                    2174       1.58    0.1%      726.6
  add_inplace           20486       0.82    0.1%       40.0
  rotate                 3643       0.77    0.1%      212.1
  subtract                165       0.27    0.0%     1607.7
  relinearize             659       0.20    0.0%      308.4
  rescale                2760       0.19    0.0%       68.8
  level_down             1336       0.08    0.0%       56.6
  conjugate               190       0.04    0.0%      229.6
  level                  6297       0.02    0.0%        3.7
  square                  197       0.01    0.0%       46.2
  noise_level              88       0.00    0.0%        5.0
  total                 80788    1154.00
```

注意：`decrypt` 的 19.74 s 是 `--per-stage` 探针的开销（258 次解密），不带 `--per-stage` 时会少很多。
`bootstrap` 每次 189 ms，22 次共 4.16 s——**bootstrap 在 GPU 上很快**。

---

## 6. 请协作者确认

1. `encode_to_light_plaintext` 是否可以预编码？`--compact` 的"惰性编码"是否可以改成"预编码"？
2. 58 ms/次是否合理？NTT on CPU for N=65536 应该在 ~1 ms 量级，58 ms 似乎有额外开销
3. 15-key 跑正在运行（test 2），完成后会有 11a1/11a2 探针数据（无 07d outlier）

Co-Authored-By: CodeAgent (GLM-5.2) <noreply@anthropic.com>
