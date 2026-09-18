# 两个优化已推，请单 batch 跑这些

日期：2026-09-18。承接 `1d70d95`。

---

## 0. 一次跑完

```bash
cd python

# A. 全量 python 测试（含新加的 3 组）
python -m pytest tests/ -q

# B. GIL：需要先重编扩展
#    cmake --build ... && pip install -e .
python -m pytest tests/test_stage2_linear.py -q        # 含 ndarray-ct subtract 的新测试

# C. 流式 QKV 的等价性和省了多少（不需要 GPU）
python -m pytest tests/test_streamed_qkv.py -q

# D. 端到端，开/关流式各一次，比时间和显存
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
  --refresh-after-dense --binary-rotations --lazy-weights \
  --scaling-bits 59 --first-mod-bits 60 \
  --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --device-memory

# E. 同上 + 流式（需要加 --stream-qkv，见下）
# F. 如果 D/E 的 headroom > 3 GiB，再试 factored：
#    加 --extra-rotation-keys 6
```

---

## 1. 优化一：释放 GIL

`python/src/bindings.cpp`，25 个计算 binding 加了
`py::call_guard<py::gil_scoped_release>()`：所有 `EvalAdd/Sub/Mult/Square/Relinearize/
MultByI/MultByInteger/Conjugate/Rotate/Rescale/LevelReduce/Bootstrap` 及其 Pt/LightPt/Scalar 重载。

**没加**的：`GetRemainingLevels`——它是个 getter，调用极频繁，
释放+重获的开销比调用本身还大。

**要重编才生效。** 正确性不受影响（守卫只改并发，不改结果），A/B 能覆盖。

---

## 2. 优化二：流式 QKV（03/04/05）

### 先说一个我核出来和预期不同的地方

「融合 03/04/05」按字面做**是亏的**。每个 `(out, diag)` 的内积要遍历**全部 64 份**
rotated copy，所以三路一起流要同时持有 `3 × 4 × 6 = 72` 个累加器，比 64 份 copy 还多。

**赢的是「按投影分别流式」**：持有 1 份 copy + 24 个累加器，代价是 copy 做三遍不共享。

### 实测（THOR 几何，depth 37，单个投影）

| | 峰值密文 | 峰值 MiB |
|---|---:|---:|
| `pcmm`（持有 copy） | 84 | 2743 |
| `pcmm_streamed` | 50 | **1652** |

**省 1.07 GiB。** 代价：copy 从做 60 次旋转变成 180 次（一层总共 8138 次）。

### 为什么正好是这个数

`bench budget --extra-rotation-keys 6` 报 headroom **2.7 GiB**，而 key generation
要 **3 GiB** 暂存——差的就是 1 GiB 左右。**这一条通了，factored key 就装得下**，
每层旋转 8138 → 4117。

### 开关

`Stages.stream_qkv`，默认 **False**。`EncoderLayer.forward` 里 `stage_02` 在开时不建数组，
三个投影各自 `iter_rotated_copies`。

**bench 还没有 `--stream-qkv` 开关**——如果你要跑 E，需要加一行：

```python
# bench.py，engine 参数组
engine.add_argument("--stream-qkv", action="store_true")
# run_encrypted，构造 layer 之后
layer.attention.stream_qkv = args.stream_qkv
```

我没加是因为不确定你那边 `run_encrypted` 有没有本地改动，怕冲突。要我加就说一声。

---

## 3. 新增的测试（都不要 GPU）

| 文件 | 测什么 |
|---|---|
| `test_streamed_qkv.py` | 流式 vs 数组：结果逐位相同（1e-12）、level 调度相同、省的字节数 >800 MiB、索引映射对得上、生成器产出和数组一致。SMALL 和 THOR 两个几何 |
| `test_stage2_linear.py` | 新增 `ndarray − ciphertext`（`he_invsqrt` 唯一走的那条 subtract 分支，之前没有任何测试）**——这条是 norm_1 那件事的关键，优先看** |
| `test_bootstrap_precision_floor.py` | Meta-BTS 的 `2^k` 增益和它的上限 |

本机 5 passed / 全套 40 passed，`pyfideslib` 相关的 skip。

---

## 4. 顺序建议

1. **B**（`test_stage2_linear.py`）——如果 `ndarray − ct` 红了，norm_1 当场结案，其它都可以等。
2. A、C——纯 CPU，几分钟。
3. D vs E——量流式在设备上省多少、慢多少。
4. F——headroom 够了再开 factored。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
