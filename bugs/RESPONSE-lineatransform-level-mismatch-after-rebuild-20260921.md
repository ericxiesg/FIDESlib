# 重编译后 `LinearTransform` level 不匹配崩溃

日期：2026-09-21。针对 `a49bae5`。**阻塞所有 GPU 实验。**

---

## 0. 结论

重编译 fideslib（从 `8fee326` 基线的旧 install 升级到 `a49bae5` HEAD）后，**所有 GPU
benchmark 在第一个 bootstrap 的 SlotsToCoeffs 阶段崩溃**：

```
python3: src/CKKS/LinearTransform.cu:76:
  Assertion `pts[0]->c0.getLevel() == ctxt.getLevel()' failed.
```

我在 `LinearTransform.cu` 加了一行 debug print，抓到了确切的 level 值：

```
[LT_DEBUG] pts0_level=36 ctxt_level=36 noise=1 pts_size=63 rowSize=63 bStep=16
[LT_DEBUG] pts0_level=35 ctxt_level=34 noise=1 pts_size=63 rowSize=63 bStep=16
```

- **第一次 LinearTransform（CoeffsToSlots）**：plaintext=36, ciphertext=36 → 通过
- **第二次 LinearTransform（SlotsToCoeffs）**：plaintext=**35**, ciphertext=**34** → 崩溃

StC 的预计算对角线在 level 35，但 ApproxModReduction 之后的 ciphertext 在 level 34——差一层。

---

## 1. 时间线

| 时间 | 事件 | install 来源 |
|---|---|---|
| Sep 16 10:00 | 旧 install 构建 | `8fee326`（41.8 MB） |
| Sep 18 14:24 | `build/fideslib.a` 重编译 | HEAD `7a03a0d`（44.6 MB）——**但没 install** |
| Sep 18 14:26 | `_core.so` 重编译 | 链接旧 install（15.6 MB） |
| **Sep 21 09:00** | `gpu_inverse_lift.log` 跑通 | **旧 install（`8fee326`）** |
| Sep 21 09:27 | 我 `cmake --install build` + 重编译 .so | HEAD（18.3 MB） |
| **Sep 21 09:32** | **崩溃** | HEAD（`7a03a0d` + `fb5d499`） |

**之前所有成功的 GPU 跑（包括 `--inverse-lift 3` 的 07d=7.33e+05 结果）用的都是 Sep 16
的旧 install（`8fee326`），不是 HEAD。** 我不知道这件事——`build/fideslib.a` 是 Sep 18
重编译的，但 `fideslib-install/` 没更新，`build-py` 链接的是 install 不是 build。

---

## 2. `8fee326`→`a49bae5` 的 C++ 改动

只有三个 commit 改了 C++（`src/` `api/` `include/`）或 `python/src/bindings.cpp`：

| commit | 改了什么 | 影响 fideslib.a？ |
|---|---|---|
| `fb5d499` | `Bootstrap` 加 `stopAfterStage` 参数（默认 -1，不改变行为） | 是 |
| `7a03a0d` | `MakeLightPlaintext` 从 2-tower 编码（你标注"待编译验证"） | 是 |
| `4c98d0b` | `bindings.cpp` 给 25 个绑定加 `gil_scoped_release` | 否（只在 .so） |

`fb5d499` 的 `stopAfterStage` 默认 -1，走原来的路径，**不应该改变行为**。
`4c98d0b` 的 GIL release 不影响 fideslib.a，但**可能引入线程安全问题**——如果 GPU
代码不是线程安全的，释放 GIL 后 Python 另起线程可能干扰 GPU 状态。

---

## 3. 我试了什么

| 实验 | 结果 |
|---|---|
| 回退 `7a03a0d`（2-tower 编码）→ 重编译 | **仍崩溃**（但可能有 stale .o 问题，见下） |
| 回退所有 C++ 到 `8fee326` → 重编译 | **仍崩溃**（`CoeffsToSlots.cu.o` 和 `LinearTransform.cu.o` 没重编译——mtime 还是 Sep 18，cmake 没检测到 header 变化） |
| 全量重编译（`touch` 所有 .cpp/.cu）→ 从 HEAD | **仍崩溃** |
| 关闭 `stream_qkv`（`--compact` 仍开） | **仍崩溃** |
| 删掉旧 plaintext cache，用全新 cold cache | **仍崩溃** |
| 回退 `bindings.cpp` 到 `8fee326` | **编译失败**——`EvalBootstrap` 签名多了 `stopAfterStage`，旧 bindings 的 `py::arg` 数量不匹配 |

**关键问题：我没能做到"从 `8fee326` 全量干净重编译"。** 回退 C++ 源码后 cmake 没重编译
依赖的 .o 文件（`CoeffsToSlots.cu.o`、`LinearTransform.cu.o` 等还是 Sep 18 编的，头文件
变了但 cmake 没检测到）。所以"回退 7a03a0d 仍崩溃"这个结论**不可靠**——二进制可能是
混合的。

---

## 4. 两个怀疑方向

### 4a. `7a03a0d`（2-tower 编码）

你自己标注了"**待编译验证**——设备上没有 CUDA，这是静态改动"。崩溃确实出现在重编译
之后。但崩溃点在 **bootstrap 的 LinearTransform**，不在 `MakeLightPlaintext`——2-tower
编码只改了 weight encoding，不改 bootstrap 对角线。

除非：bootstrap precomputation 内部也调了 `MakeLightPlaintext` 或类似的 encoding 路径，
而那个路径的 level 计算被 `cheapestLevel` 改了。**需要你确认 StC 对角线是用什么编码的。**

### 4b. `4c98d0b`（GIL release）

25 个绑定加了 `py::call_guard<gil_scoped_release>()`。如果 GPU 代码有**非线程安全的
全局状态**（比如 context pool、level bookkeeping），释放 GIL 后 Python 的 GC 或其他
线程可能跑进来破坏状态。

这能解释为什么**只差一层 level**——不是完全随机，而是某个计数器被并发访问 off-by-one。

**验证方法**：把 `bindings.cpp` 里 `py::call_guard<py::gil_scoped_release>()` 全部删掉
（保留 `stopAfterStage` 的 `py::arg`），重编译 .so，看崩溃是否消失。这不影响 fideslib.a。

---

## 5. 我现在卡住的地方

1. **没法从 `8fee326` 干净重编译**——cmake 的依赖追踪没覆盖 header 变化，回退源码后
   .o 文件不重编译。需要 `rm -rf build && cmake -S . -B build ...` 全重来，但编译要
   ~5 分钟。

2. **没法单独回退 `bindings.cpp`**——`8fee326` 的 bindings 没有 `stopAfterStage` 的
   `py::arg`，和 HEAD 的 `CryptoContext.hpp` 签名不匹配。需要手动合成一个"有 stopAfterStage
   但没 GIL release"的版本。

3. **`--inverse-lift 3` 的 07d 结果（7.33e+05）是用旧 install 跑的**，不是 HEAD。如果
   协作者有 C++ 改动依赖这个结果，需要注意它来自 `8fee326` 基线。

---

## 6. 需要协作者看的事

1. **`7a03a0d` 的 2-tower 编码是否影响 bootstrap 对角线的 level？** 崩溃在 StC 的
   `LinearTransform`，plaintext level 35 vs ciphertext level 34。StC 对角线是怎么编码的？
   `cheapestLevel = towers - 2` 是否改变了它们的 level？

2. **`4c98d0b` 的 GIL release 是否安全？** GPU context 的 level/scale bookkeeping 是否
   线程安全？能否给一个"有 `stopAfterStage` 但没 `gil_scoped_release`"的 bindings.cpp
   让我测？

3. **是否需要我从 `8fee326` 全量干净重编译（`rm -rf build`）来确认旧基线确实不崩？**
   这能排除"build 环境变化"（CUDA 12.9→13.3 等）的因素，但编译要 5 分钟。

4. **`--inverse-lift 3` 的探针数据（07d=7.33e+05, 11b=2.08e+154）来自旧 install。**
   协作者需要的 `inv_iter01_b` 到 `inv_iter03_b` 逐槽数据我**还没跑到**——崩溃在
   bootstrap 阶段，在 he_inv 探针之前。

---

## 附：环境

- GPU: Quadro GV100 32GB (sm_70)
- CUDA: `/usr/local/cuda` → `cuda-13.3`（但 cmake cache 里 `CUDA_PATH=cuda-12.9`）
- fideslib install: `/home/zhiyuan/workspace/THOR-v2/fideslib-install`（44.6 MB，HEAD）
- Python: `/home/zhiyuan/.pyenv/versions/3.11.0/bin/python3`
- git HEAD: `a49bae5` (bootstrap-dev)
- 构建配置: Release + `-DNDEBUG -UNDEBUG`（asserts 已恢复）
- 崩溃命令（最简）:
  ```bash
  PYTHONPATH=... python3 -m thorfhe.bench fhe --engine fideslib --device cuda:0 \
    --depth 37 --dnum 4 --bootstrap-level-budget 3,3 --binary-rotations \
    --refresh-after-dense --residual-scale 256 --refresh-scale 4 \
    --score-refresh-scale 16 --layers 1 --limit 1 --compact \
    --extra-rotation-keys 6 --rotation-max-steps 4 --inverse-lift 3
  ```
