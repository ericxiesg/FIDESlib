# batch_size = 1 下的优化评估：Part17 十项逐条，以及 C++/Python 取舍

日期：2026-09-15。凡标 **[实测]** 的是在本机 `ClearEngine` 上跑出来的；标 **[待远端]** 的
是需要 GPU 才能定的数。参照：`Part17-FIDESlib-vs-EasyFHE-Python对接架构对比.md`。

---

## 0. 三条结论

1. **模型继续留在 Python。** Python 侧开销占全程 1% 以下（下面第 1 节有数）。
   把模型搬进 C++ 买到的就是这 1%，卖掉的是双后端对照——这个项目至今最有效的
   查错手段。
2. **最大的一笔优化不在语言层，在旋转基。** 多 4～6 把旋转钥匙，一层的旋转次数从
   8138 降到 4413／4117（−46%／−49%），显存还剩 0.83／0.52 GiB。旋转按估算占
   GPU 时间八成以上。**已实现**（第 2 节，预测与实跑逐格相符）。
3. **Part17 十项里只有 ⑧（batch 维）在 batch_size=1 下真的失效。** 其余九项里，
   报告中被称作"批处理"的收益大多不是**样本批**，而是*同一个样本内部的密文批*
   （4/8/16/64 个）和*同一个密文的多次旋转批*（`make_copies` 一次 16 个）——
   这些在 batch_size=1 下**照样成立**。

---

## 1. C++ 还是 Python

### 1.1 先澄清一个前提：OpenFHE 后端已经存在

`ClearEngine` 不是 CKKS 的模拟替身，它是**第二个后端**：

| | 后端 | 用途 |
|---|---|---|
| `pyfideslib.Engine` | `api/CryptoContext` → `lbcrypto::CryptoContext<DCRTPoly>`（OpenFHE，负责 keygen / encode / encrypt / decrypt / bootstrap setup）+ FIDESlib CUDA（负责 evaluation） | 真跑 |
| `ClearEngine` | numpy，精确复数槽 | 对照 + FIXEDMANUAL 类型检查（level / scale / degree） |

`bench.py:113` 只有显式 `--engine clear` 才走 numpy，否则 `bench.py:151` 构造
`pyfideslib.Engine`。`src/bindings.cpp` 已绑 93 个方法。

`Stages` 只写一遍，两边都能跑——"移植对不对"因此化简成"两次运行对不对得上"。
这一条已经付清了它的成本：softmax 那个 bug 我连猜错三次方向（EvalNegate 的 scale、
噪声底、bootstrap），每次都是这条对照链把我纠回来。FHE 的错误形态是"解密出一个
看起来合理的错值"，不是崩溃，没有对照就只能猜。

### 1.2 Python 占多少 [实测]

一层的引擎原语调用（`binary_rotations=True`，`refresh_after_dense=True`）：

| 原语 | 次数/层 |
|---|---:|
| multiply | 23308 |
| add | 22666 |
| add_inplace | 20486 |
| **rotate** | **8138** |
| rescale | 2764 |
| level_down | 1336 |
| relinearize | 665 |
| conjugate | 196 |
| multiply_1j | 174 |
| subtract | 168 |
| bootstrap | 22 |
| **合计** | **79923**（×12 层 = 959076） |

pybind11 一次往返约 1–3 µs，96 万次合计 **1–3 秒**。而单次旋转在 N=2¹⁶、L≈20 的
V100 上是毫秒量级，8138 次就是每层数十秒。**Python 占比 < 1%。**

需要 C++ 的只有一类：**跨原语融合**（比如把 `attention._accumulate` 内层的
rotate→multiply→rescale→mask 合成一次调用，省 kernel launch 和中间显存）。
那是按需下沉单个热点循环，不是全量重写，而且下沉之后 `ClearEngine` 只要实现同一个
复合原语就仍能对照。

---

## 2. 旋转基：本次实现的主要优化

### 2.1 问题

一层要 210 个不同的旋转索引。深度 37 下截断后 **27.85 GiB**，总驻留 53.67 GiB——
32 GB 卡装不下。现行对策是只留 2 的幂（15 把钥匙，2.52 GiB），任意索引拆成置位数
之和：**1802 次旋转变成 8138 次**（4.5×）。

### 2.2 观察 [实测]

索引不是任意的。最重的一处是 `attention.py:383`（`_accumulate`），对角线 `in_index`
的旋转量是

```
rotation = g.group_size * j - g.n_slot * in_index,   in_index = pack * block + j
```

`pack=16`、`group_size=2048`、`n_slot=16` 代入即 **`rotation = 2032*j - 256*block`**——
j∈[0,16)、block∈[0,8) 的完整二维网格。那 127 个索引每一个都是两个基元素之和，
二进制要八步，两步就够。

同源旋转的统计（用强引用防止 `id()` 复用后重测）：

| | 旋转总数 | 同源可 hoist | 分组直方图 |
|---|---:|---:|---|
| 一索引一钥匙 | 1802 | 31.6% | 2×127 + 4×63 + 3×16 + 1×15 |
| 二进制 | 8138 | 7.0% | 同上（只有链首共享） |

（先前记的 82.8%／96.5% 是 `id()` 被 GC 复用造出来的假象，已作废。）

### 2.3 做法

不把上面那个代数式写死——几何一变就会悄悄失效——而是**测**：拿一次 dry run 真正
请求过的索引和频次，贪心地加"能省下最多旋转"的那把钥匙，加到卡装得下为止。
`thorfhe/rotation.py`。

### 2.4 结果 [实测]

深度 37、dnum 4、11 个特殊素数、截断：

| 额外 key | 总 key | 旋转次数/层 | 比二进制 | 旋转钥匙 | 总驻留 | 32 GB 余量 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 8138 | 100% | 2.52 G | 28.34 G | 1.46 G |
| 2 | 17 | 4706 | 58% | 2.82 G | 28.64 G | 1.16 G |
| **4** | 19 | **4413** | **54%** | 3.15 G | 28.97 G | **0.83 G** |
| **6** | 21 | **4117** | **51%** | 3.46 G | 29.28 G | **0.52 G** |
| 9 | 24 | 3667 | 45% | 3.93 G | 29.75 G | 0.05 G |
| 10 | 25 | 3520 | 43% | 4.09 G | 29.91 G | **−0.10 G 装不下** |

候选池是**实际用过的那 210 个索引**，不是所有 2032j/256b 的倍数。放开候选池在第 9 把
钥匙上能再省 0.4%（3667→3654），不值得：那会挑出一把只在链中间出现、没有哪个 stage
单独用到的钥匙，而钥匙集与代码的对应关系是这里唯一能靠眼睛查的东西。

建议默认 **`--extra-rotation-keys 4`**（省 46%，留 0.83 GiB）；
`6` 是激进档（省 49%，留 0.52 GiB）。驻留是 `budget.estimate` 的**预测值**、
含标定过的 overhead 项，所以 0.5 GiB 余量偏薄，**要在远端用 `--device-memory` 复核**。

### 2.5 端到端验证 [实测]

预测的旋转次数必须等于真跑出来的次数，且真跑时请求的每一个索引都必须在规划的钥匙集里——
后者是那条"缺钥匙不报错"的实际检查：

| 配置 | 预测旋转 | 实跑旋转 | keys | 驻留 | 余量 | 钥匙齐全 |
|---|---:|---:|---:|---:|---:|---|
| binary | — | 8138 | 15 | 28.34 G | 1.46 G | 是 |
| binary+2 | 4706 | 4706 | 17 | 28.64 G | 1.16 G | 是 |
| binary+4 | 4413 | 4413 | 19 | 28.97 G | 0.83 G | 是 |
| binary+6 | 4117 | 4117 | 21 | 29.28 G | 0.52 G | 是 |
| binary+9 | 3667 | 3667 | 24 | 29.75 G | 0.05 G | 是 |

### 2.6 与 bootstrap rescale 的耦合 [实测]

上面那张表是 depth 37 的。`72dc818` 在 FIXEDMANUAL 下给 bootstrap 补了一次 rescale
（`EvalBootstrap` 返回 NoiseLevel=2，不补就指数放大——见
`bugs/RESPONSE-bootstrap-noise-level-20260915.md`），这多花一格 level。
两次 bootstrap 之间最深的一段要 19 格，所以 depth 37 可能不够：

```
depth  bootstrap落点  结果
   37            20  OK   出口 level 1
   37            19  FAIL rescale: would leave the ciphertext at level -1
   38            20  OK   出口 level 1
```

depth 38 每把钥匙多一层 limb，吃掉 0.44 GiB 余量（旋转次数不变，基的选择与 depth 无关）：

| depth | 额外 key | 旋转/层 | 总驻留 | 32 GB 余量 |
|---:|---:|---:|---:|---:|
| 37 | 4 | 4413 | 28.97 G | 0.83 G |
| 37 | 6 | 4117 | 29.28 G | 0.52 G |
| 38 | 4 | 4413 | 29.42 G | **0.38 G** |
| 38 | 6 | 4117 | 29.74 G | **0.06 G** |

**要不要上 38，取决于 `EvalBootstrap` 本身吃 16 格还是 17 格**，这个数还没在 bench
的配置下量过。真上 38 的话，额外钥匙的上限从 +9 掉到 +4，`--extra-rotation-keys 4`
就同时是建议值和天花板。

### 2.7 正确性

分解的充要条件是 `sum(steps) ≡ index (mod slot_count)`。`tests/test_rotation_basis.py`
对全部 32768 个索引穷举验证了这一条（三种基），另外验证了：

* 每一步都落在基里（不会去要一把没建的钥匙）；
* 分解与钥匙的给定顺序无关——否则**规划出的钥匙集和实际花掉的钥匙集会不一致**，
  而缺钥匙不报错，它表现为解密出一个无关的值；
* `key_levels` 给出的 level 覆盖每一步。

规划与执行共用同一个 `RotationBasis.steps`，这是上面那条不一致的结构性防线。

### 2.8 改了哪些文件

| 文件 | 改动 |
|---|---|
| `python/thorfhe/rotation.py` | 新增：`RotationBasis`、`factored_basis`、`key_levels`、`rotation_cost` |
| `python/thorfhe/clear.py` | 新增 `rotation_counts`（频次；原有的 `rotation_levels` 只说要哪些钥匙，频次说哪把值得花） |
| `python/thorfhe/stages.py` | `binary_rotations` 放宽为 `bool \| RotationBasis`；`rotation_steps` 委托给基 |
| `python/thorfhe/he.py` | 新增 `plan_rotations` → `RotationPlan(levels, basis, rotations)`；`plan_rotation_keys` 变成取 `.levels` 的薄包装 |
| `python/thorfhe/bench.py` | `--extra-rotation-keys N`（三个子命令）；`rotation_mode()` 让 layer 拿到规划用的那个基 |
| `python/tests/test_rotation_basis.py` | 新增 15 个用例 |

---

## 3. Part17 十项逐条（batch_size = 1）

| # | 项 | batch=1 是否成立 | 评估 | 优先级 |
|---|---|---|---|---|
| ① | 绑 `EvalFastRotation*`（hoisting） | **成立** | 批的是*同一密文的多次旋转*，与样本批无关。C++ 已有批量重载 `EvalFastRotation(ct, vector<indices>, m, precomp)`（`api/CryptoContext.hpp:269`），未绑。**[实测]** 一索引一钥匙下 31.6% 的旋转同源；二进制下只有 7%（链首）。换成 2.4 的因子基后，两步链的第二步同源比例会升高，值得重测 | **中高**（等 2.4 落地后重估） |
| ② | 全部 `.def` 加 `py::call_guard<gil_scoped_release>()` | 成立 | `src/bindings.cpp` 里目前**一个都没有**（grep 零命中）。单线程下不直接省时间，但它是任何 CPU/GPU 重叠（预编码、light plaintext 展开）的前提。改动机械、无逻辑风险 | **中**（便宜的保险） |
| ③ | 绑 in-place 变体 | 成立 | 最热的一个已经是真 in-place：`pyfideslib/__init__.py:135` → `EvalAddInPlace`，覆盖 20486 次/层。**缺的是 `rescale`（2764 次/层）和 `multiply`**——每次多一份 L 层密文的分配与拷贝 | **中** |
| ④ | 暴露 `mult` 的 rescale/moddown 融合 | 成立 | 23308 次 multiply，每融合掉一趟就是一遍全密文的读写 | **中** |
| ⑤ | light plaintext 缓存容量 | 成立 | Part17 说我们默认 8，**实际是 64**（`pyfideslib/__init__.py:50`）。Python 侧另有一个按内容 key 的无界字典（`stages.py:169`），所以 encode 只做一次；64 这个数管的是**按 level 展开**的缓存。一层要多少个 (明文, level) 组合我还没量 | **中**，先量再调 **[待远端]** |
| ⑥ | 旋转钥匙常驻 CPU、按 level 裁剪上传 | 成立 | 这是 2.4 那张表的**上限解除器**：现在 +10 把钥匙就装不下，能常驻 CPU 就能继续往 3667 以下走。但单把钥匙百余 MiB，PCIe 上传约 10 ms，和一次旋转同量级——**得先量 PCIe 带宽和钥匙复用间隔再定** | **低**，除非要突破 +9 **[待远端]** |
| ⑦ | Python 侧 level/scale 记账 | 成立 | **已具备**：`ClearEngine` 全程记 level/scale/degree，`GetNoiseLevel` 也已加。剩的是把它挂到真引擎路径上做断言 | **已完成大部** |
| ⑧ | batch 维 | **不成立** | 唯一真正失效的一项。batch_size=1 就没有样本维可批 | **放弃** |
| ⑨ | GPU 上 encode | 成立但价值低 | Python 侧内容缓存已经把重复 encode 消掉了，剩下的是首层的一次性成本 | **低** |
| ⑩ | bootstrap 契约（`output_state` + context 指纹） | 成立 | 与性能无关，与**正确性**有关。我们已经在 bootstrap 落点上栽过：`bench.py` 里现在是从 `depth - GetBootstrapDepth()` 推出来而不是设进去的，就是这个教训。把契约显式化能防下一次 | **中**（正确性项） |

---

## 4. 建议顺序

1. **`--extra-rotation-keys 4`**（已实现）——远端用 `--device-memory` 复核驻留，
   确认 0.83 GiB 余量是真的，再考虑 `6`。
2. **绑 `EvalFastRotation` 批量重载**（①）——先在新基下重测同源比例。
3. **`rescale` / `multiply` 的 in-place 变体**（③）+ **mult 融合**（④）。
4. **`gil_scoped_release`**（②）——机械改动，顺手做掉。
5. **量 (明文, level) 组合数**（⑤），再决定 `light_plaintext_cache`。
6. ⑩ 的 bootstrap 契约。
7. ⑥ 只在确实要突破 +9 把钥匙时再碰。

⑧ 放弃。⑦ 只补真引擎路径上的断言。
