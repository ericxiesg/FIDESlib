# 回复 GPU-OOM-210keys-20260907

日期：2026-09-07。本机没有 GPU，下面的显存数字是用你日志里的实测值标定出来的模型，代码改动跑过本机测试。

先说好消息：`dnum=4` 让 `multiply_1j` 在 depth=50 通过，**证实了 MAXP 溢出的诊断**——之前那三个
「不同的 bug」确实是同一个越界。109 个 pytest 全过也说明 Python 侧的原语对齐是好的。

坏消息是 210 把 key 装不下，而且**不是靠截断能解决的**。下面是算清楚的账，以及一个能装下的办法。

---

## 一、截断救不了：即使 100% 生效也要 40 GiB

先标定单 key 大小。你日志里 `Rotation keys loaded: 49 ~ 12152MB` → 248 MiB/key。按
`full_key = 2 * dnum * (L + 1 + K) * N * 8`，代入 dnum=4、N=2^16、L=50 解出 **K = 11**，
所以每 key 是 62 个 limb-tower。

用 `plan_rotation_keys(depth=50, bootstrap_level=40)` 的真实 level 分布算：

| | |
|---|---:|
| 210 把 key，不截断 | **50.9 GiB** |
| 210 把 key，按 plan 完美截断 | **40.3 GiB**（只省 21%） |
| + bootstrap 明文 10.4 GiB + bootstrap key 11.9 GiB | **62.6 GiB** vs 32 GiB |

为什么截断这么没用——看 level 直方图：

```
level  0- 9:   1 key
level 20-29:  66 keys
level 30-39:   3 keys
level 40-49: 140 keys     <- 三分之二的 key 用在接近满 level 的地方
```

那 140 把是 **stage 01–05 的旋转**：它们发生在第一次 bootstrap *之前*，密文还是满 level，
所以 `maxLevel >= L`，`api/CryptoContext.cpp:255` 那行直接给 `-1`（不截断）。截断机制没坏，
是这个负载本来就没什么可截的。

所以你日志里的 `0 truncated` 是**正确行为**，不是 bug。

---

## 二、能装下的办法：把旋转拆成 2 的幂，210 把 → 15 把

旋转在槽空间里是**可加的**，而且**不消耗 level**。所以任何索引都能用 2 的幂拼出来，
`slot_count = 2^15` 意味着基只要 15 把 key。

已实现：`Stages.binary_rotations`（默认关，`EncoderLayer(..., binary_rotations=True)`、
`plan_rotation_keys(..., binary_rotations=True)`、`bench --binary-rotations` 都能开）。

实测同一层：

| | key 数 | 不截断 | 截断后 | 每层旋转次数 |
|---|---:|---:|---:|---:|
| 现在 | 210 | 50.9 GiB | 40.3 GiB | 1802 |
| `--binary-rotations` | **15** | **3.6 GiB** | **3.3 GiB** | 8138 |

**代价是 4.5 倍的旋转次数换 14 倍的 key 显存。** 加上 bootstrap 的 22.3 GiB，总共约
**25.9 GiB**，在 32 GiB 卡上装得下。

正确性：`rotate` 的结果和直接旋转**逐位相同**（旋转是精确置换，拆开只是多做几次），
`test_gpu_resource_fixes.py` 里两个用例钉住了这一点。唯一的实际代价是每次多几层 key-switch 噪声。

### 中间档（还没做，但数据都在）

4.5 倍全上有点狠。可以给最常用的少数索引留专用 key、其余拆开——1802 次调用里绝大多数集中在少数几个
索引上（stage 02 的 ±2048、`rotate_internal` 的 ±1..±12）。给 20 把专用 key 大概能把旋转次数拉回
2 倍出头而显存只多 5 GiB。要做的话把 `Stages.rotation_steps` 改成查一个基集合就行，接口已经留好了。

---

## 三、请你下一轮跑这个

```python
plan = plan_rotation_keys(THOR_BERT, depth=50, bootstrap_level=40, binary_rotations=True)
# 15 个 index

e = pyfideslib.Engine("cuda:0", log_n=16, depth=50, scaling_bits=50, first_mod_bits=55,
                      dnum=4, rotation_indexes=plan,
                      bootstrap_level_budget=(3, 3),
                      secret_key_dist=pyfideslib.SPARSE_TERNARY)
```

然后

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 --per-stage --binary-rotations
```

`--per-stage` 会逐级解密并报 best-fit scale + 残差，CPU 侧的参考值在
`docs/thor_port.md`（query/value/scores 应该是 scale 2.0 / 残差 ~3e-7，softmax 1.0023 / 2.3e-6，
层输出 relRMSE ~1.1e-3）。GPU 上会在这之上叠 CKKS 噪声，但 scale 那一列应该完全一样——
**哪一级的 scale 先偏掉，就是那一级的问题**。

## 四、关于你的四条建议

1. **按需加载 key** —— 同意，这是通用解，但工作量不小。先用 `--binary-rotations` 顶一下，
   15 把 key 全驻留只要 3.6 GiB，比实现 lazy-load 便宜得多。
2. **让截断生效** —— 见第一节，截断已经生效了，是这个负载没什么可截的（140/210 用在满 level）。
   `0 truncated` 是对的。
3. **bootstrap key 和 rotation key 共享 aux 池** —— bootstrap 那 10.4 GiB 是**明文**（StC/CtS 矩阵），
   和 rotation key 的 LimbPartition 不是一回事，共享池帮不上。真要压它，`levelBudget={4,4}`
   会减少线性变换的 giant-step 数，明文和 key 一起降——这是 TODO 里的 T7，值得单独试一次。
4. **降 log_n** —— 不行，THOR_BERT 的 packing 就要 32768 槽。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/stages.py` | `binary_rotations` + `rotation_steps`（2 的幂分解） |
| `python/thorfhe/attention.py` | `AttentionStages.__init__` 透传 `**kwargs` |
| `python/thorfhe/layer.py` | `EncoderLayer(..., binary_rotations=)` 透传到四个 stage 对象 |
| `python/thorfhe/he.py` | `plan_rotation_keys(..., binary_rotations=)` |
| `python/thorfhe/bench.py` | `--binary-rotations` |
| `python/tests/test_gpu_resource_fixes.py` | 两个用例：分解精确、`rotation_steps` 就是二进制展开 |

你那两处 debug print 的删除保留了。
