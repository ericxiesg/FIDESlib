# 回复 v6：先别 revert，那个前提不成立

日期：2026-09-11。本机仍无 GPU，**C++ 一行没编译过**。

先说重点：**一层跑通了，这是项目第一次**。stage 06 之后的全部 stage 第一次在 GPU 上走完，
24 分钟 4 个样本。恭喜。

然后是三件需要纠正的事，其中第一件会直接改变下一步该做什么。

---

## 一、没有「回归」可言——stage 11 之后从来没有跑对过

v4 是**挂在 stage 11** 的（`he_invsqrt` 的 `TypeError`）。所以：

| stage | GPU 上验证过吗 |
|---|---|
| 01–05 | 过，relRMSE **1.03e-8** |
| 06–10 | 跑完过，但**从未逐级测过保真度** |
| 11–16 | **从来没有跑完过** |

「revert 三个 C++ 改动逐个二分」这个计划的前提是**曾经有一个正确的整层结果**，
现在退回去能找回它。**没有这个基线。** relRMSE 1.0 完全可能不是任何一个 C++ 改动引入的，
而是 06–16 里某个从来没在 GPU 上验证过的 stage 本来就是错的。

四次 GPU 跑（每次 24 分钟起）去二分一个可能根本不存在的回归，我建议不要做。

## 二、`dc2ef1d` 不是「每个密文泄漏它的 context」

这条影响你的判断，所以要说清楚：那个修复是**context 本身被 drop 时不释放显存**——
`KeySwitchingKey` 持有 `shared_ptr<ContextData>`，而 key 就活在 `ContextData::precom.keys` 里面，
成环。**一次运行只建一个 context**，所以它不可能解释 150 分钟 → 24 分钟。

真正让它变快的是 v4/v5 之间那几轮工作集削减：`forward` 里中间结果最后一次使用后立即释放
（峰值 5.43 → 3.43 GiB）、`make_copies` 改成流式（stage 06 3.43 → 2.18 GiB）、辅助池 trim。
泄漏修复对**长期跑多个 context 的进程**才有意义，对 benchmark 是零。

## 三、对你列的三个嫌疑，我的判断

* **`LimbPartition::multPt` 用 `limb.at(limbsize-1)`**：只在 `limb.size() != limbsize` 时改变结果，
  而且是**改对**。`dropToLevel` 不缩物理存储，旧代码 `back()` 取的是当前 level 之外那根 limb。
  light plaintext 反复在不同 level 上展开同一个池化多项式，是最容易踩到的场景。
  如果 revert 它「修好」了保真度，那说明**有别的代码依赖着取错 limb**，那才是真 bug。
* **`rotate_hoisted` 拷 `slots`**：旧行为下 hoisted 结果的 `slots` 是 0，而
  `GetRotationKey` 只有在**精确 index 不存在**时才会走模-slot 回退（`i < N/2/slots`，slots=0 就是除零）。
  我们用二的幂旋转，15 个 index 全都精确存在，那条回退从来不触发。**所以这条对我们应该是零影响。**
* **weak_ptr 泄漏修复**：只改了 key 怎么拿到 context，不碰任何数值。

三个里没有一个能自然地解释「输出与明文完全不相关」。

---

## 四、改做什么：一次 `--per-stage`，不是四次 revert

`--per-stage` 以前之所以撑爆显存，是因为 **trace 把每个 stage 的输出密文一直攥到整层结束**——
那是整层工作集的好几倍。已经改掉了：

`EncoderLayer.trace_sink` —— 有 sink 时，**在 stage 边界当场解密**，trace 里留下的是一个 numpy
数组而不是一堆活着的密文。附带好处：stage 08 的 `consume` 现在**带 trace 也能开**
（以前必须关掉，因为 trace 要留着 softmax 对角线），所以 `--per-stage` 不再比普通跑更费显存。

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
    --binary-rotations --refresh-after-dense --depth 37 --dnum 4 \
    --bootstrap-level-budget 3,3 --light-plaintext-cache 4 \
    --per-stage --device-memory
```

会打出这张表（下面是本机 clear engine 的值，GPU 上应该接近）：

```
  query            scale 2.0000  relRMSE 2.9e-07
  scores           scale 0.5000  relRMSE 2.9e-07
  softmax          scale 1.0023  relRMSE 2.3e-06
  attention_dense  scale 2.0044  relRMSE 3.0e-04
  norm_1           scale 1.9998  relRMSE 6.5e-04
  intermediate     scale 0.0313  relRMSE 4.8e-04
  gelu             scale 2.0001  relRMSE 2.3e-03
  output_dense     scale 2.0000  relRMSE 2.2e-03
  norm_2           scale 1.9998  relRMSE 1.1e-03
```

**要看的就一件事：从哪一行开始 `scale` 塌到 0、relRMSE 跳到 1。** 那一行就是答案，
一次跑就够，不用 revert 任何东西。

第 1 步「用 clear engine 跑 `--per-stage` 验证 Python 管线」可以跳过——上面这张表就是 clear engine
刚跑出来的，Python 侧是对的，91 个用例也全过。再跑一遍只会确认已知的事。

## 五、如果表指向 stage 11

那我先怀疑 `he_invsqrt`。它是 Goldschmidt 迭代，**输入落在收敛域外就会发散而不是变不准**。
精确算术下收敛，CKKS 噪声下如果输入贴着边界就可能炸——这正好能解释「输出与明文完全不相关」
而不是「输出有点偏」。表里 `norm_1` 那一行会直接说明是不是它。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/layer.py` | `EncoderLayer.trace_sink`；stage 08 的 `consume` 只在「trace 且无 sink」时关闭 |
| `python/thorfhe/bench.py` | `--per-stage` 装上 sink，在 stage 边界解码；`per_stage_fidelity` 接受已解码的数组 |
