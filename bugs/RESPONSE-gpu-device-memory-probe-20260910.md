# 追加：给下一跑加一个 `--device-memory`

日期：2026-09-10，接在 [RESPONSE-gpu-workingset-measured-20260910.md](RESPONSE-gpu-workingset-measured-20260910.md) 后面。
**C++ 改动仍然一行都没编译过**，改的地方都很小，但请留意。

## 为什么加这个

你这轮的结果里有一条我的模型解释不了，而它比 OOM 本身更值得追：

| stage | 主机侧模型算出的工作集 | GPU 上 |
|---|---:|---|
| stage 06 | **5.43 GiB** | **过了** |
| stage 07 | 4.58 GiB | **挂了** |

大的过了、小的挂了。**说明真正卡住的东西不在我的模型里**——模型只数活着的密文，
数不到显存池自己攥着多少。继续按模型削减工作集，就是在猜。

而且到现在为止，每一次改动的验证信号都只有一个比特：**「这一跑走得比上一跑远吗」**，
一跑 90 分钟。这个比特把两种完全不同的情况混在一起了：

* 池子攥着一堆没人用的块（`pooled - in_use` 很大）→ **电路不是问题，回收才是**；
* `driver_free` 很小而 `pooled` 也不大 → **是密钥和明文占满了**，削电路没用。

## 加了什么

`FIDESlib::GetPoolStats(id)`，四个数：`pooled`（池子从 driver 拿走且从没还过的字节）、
`in_use`（其中已经发出去的，= pooled 减各 size class 空闲表）、`driver_free`、`driver_total`。
`in_use` 是**推出来的不是数出来的**——池子拥有的每一块要么发出去了要么在空闲表里，
所以相减就是精确值，分配快路径上不需要加任何记账。

一路透出到 `CryptoContextImpl::GetDeviceMemory()` → `Engine.device_memory()` →
`Stages.device_memory()`（无设备时返回 `{}`）→ `EncoderLayer.memory_probe`，
**在每个 stage 边界、`trim` 之后**打一行。之后读而不是之前读是有意的：那才是下一个 stage
真要分配时还被占着的量。

## 下一跑

在上一份给的命令后面加 `--device-memory`：

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
    --binary-rotations --refresh-after-dense --depth 37 --dnum 4 \
    --bootstrap-level-budget 3,3 --light-plaintext-cache 4 --device-memory
```

会看到这样的行：

```
  [mem] stage_06_attention_score          pool 12.30 GiB (in use  9.10, reclaimable  3.20)  driver free  2.10 GiB
```

**就算这一跑跑通了，也请把这些行发回来**——它们是第一份真实的显存去向数据，
比「过了没过」有用得多。跑挂了就更要发，挂之前最后几行直接指出该削哪儿。

顺带：`stage 07` 那个 OOM，上一份说的结论不变——你跑的 560e54d 上 stage 07 的工作集是
4.58 GiB，`drop()` 之后是 2.34 GiB，大概率已经解决了。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CudaUtils.{cuh,cu}` | `PoolStats` / `GetPoolStats(id)` **未编译** |
| `api/CryptoContext.{hpp,cpp}` | `GetDeviceMemory()`，CPU 上下文返回全 0 **未编译** |
| `python/src/bindings.cpp` | 绑定 `GetDeviceMemory` **未编译** |
| `python/pyfideslib/__init__.py` | `Engine.device_memory()` |
| `python/thorfhe/stages.py` | `Stages.device_memory()`，无设备时 `{}` |
| `python/thorfhe/layer.py` | `EncoderLayer.memory_probe`，stage 边界 trim 之后回调 |
| `python/thorfhe/bench.py` | `--device-memory` |
