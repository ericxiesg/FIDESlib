# 回复 GPU-benchmark-bugs-20260907

日期：2026-09-07。**本机没有 GPU，下面 C++ 侧的改动一行都没编译过**，Python 侧全部跑过测试。

先说结论：报告很准，六个都成立。但 Bug 1/3/5 的根因和报告里的怀疑方向不一样，Bug 6 的修复引入了一个
新问题。逐条如下。

---

## Bug 1（multMonomial depth≥50 NaN / ≥58 segfault / ≥63 创建失败）：根因是 `MAXP` 溢出，不在 multMonomial

`src/ConstantsGPU.cuh:12` 有

```cpp
constexpr int MAXP = 64;
```

GPU 常量结构里**每一张表**都是 `[MAXP]`，而且 `MAXP` 同时是若干扁平指针表的**步长**
（`ElemenwiseBatchKernels.cu` 里的 `w[i * MAXP + primeid]`、`LimbPartition.cu` 里的
`4 * MAXP`、`(4 + DECOMPmeta.size()) * MAXP` 等等）。

`src/ConstantsGPU.cu` 填表时：

```cpp
hC_.L = q.size();
hC_.K = p.size();
for (i < q.size())  hC_.primes[i]        = q[i].p;
for (i < p.size())  hC_.primes[hC_.L + i] = p[i].p;   // 最高写到 L + K - 1
```

所以约束是 **`L + K <= MAXP`**，而**没有任何地方检查**。越界不会报错：`primes[64]` 写进相邻的
`prime_better_barret_mu[0]`，再往上依次污染 `N_shoup`、`root`、`primeid_flattened`……

这正好解释报告里那三个阈值——它们不是三个 bug，是**同一个溢出越走越远**：

| depth | 溢出深度 | 表现 |
|---|---|---|
| ≥ 50 | 刚越过 `primes`，污染 Barrett/Shoup 常数 | 模乘结果错 → **NaN**，无异常 |
| ≥ 58 | 够到 `primeid_flattened` / 指针表 | 索引成野指针 → **illegal memory access** |
| ≥ 63 | 越过整个结构体 | `cudaMemcpy` **'invalid argument'** |

也解释了为什么 `first_mod_bits=60` 会更早崩：首模更大 → `computeK` 需要更多特殊素数 → K 更大 →
更早越过 64。

`multMonomial` 本身没问题，它只是第一个用到被污染常数的原语（`EvalMultByI` 走
`multElement` → `Mult_` kernel，直接吃 `C_.primes` 和 Shoup 常数）。

### 已改

`src/ConstantsGPU.cu` 在 `SetupConstants` 入口加了检查，把静默污染变成带数字的异常：

```
FIDESlib: this parameter set needs <q> ciphertext primes + <p> special primes = <total>,
but the GPU constant tables hold MAXP = 64. Lower `depth`, or raise `dnum` (which shrinks
the special-prime count K), or raise MAXP in src/ConstantsGPU.cuh and rebuild - note MAXP
costs O(MAXP^3) constant memory through DecompAndModUp_matrix, so raising it is not free.
```

**请你重跑一次 depth=50 和 depth=58，把这条异常里的 q / p / total 三个数字发回来。** 我这边算不出
K（`computeK` 依赖每个 digit 的实际 logQ 和特殊素数位宽），拿到真实数字才能给出准确的 depth 上限表。

### 建议的下一步（按性价比）

1. **先试 `dnum=4` 或 `5`**。K ≈ 最大 digit 的素数个数，`dnum` 越大 K 越小，所以同样的 depth 下
   `L + K` 会掉下来。这不需要改任何 C++，可能立刻解掉 Bug 1 和 Bug 3 的一半。
2. 真要 depth ≥ 50，把 `MAXP` 提到 72 或 80 重编。代价是 `DecompAndModUp_matrix` 是
   `MAXP^3 * MAXD` 字节：64 → 2 MB，80 → 4 MB，每 GPU 两份（本体 + shoup）。可以接受，但先做 1。
3. 更根本的是**不需要 depth 50**：带 bootstrap 的话 depth ~30 就够。也就是说 Bug 1/3 都是被 Bug 2
   逼出来的。

---

## Bug 3（pcmm OOM）：已修，峰值降到 1/out_dim

报告说得对。`pcmm` 之前先调 `parallel_diagonal_pc_mult` 把整个 `(out_dim, diag_dim)` 网格算出来，
然后才逐行折叠——但**每个输出密文只需要自己那一行，而且只需要它活到被旋转累加为止**。

已改成边算边折：每行算一个对角块就立刻 `rotate_internal` 累加进去，同时活着的临时密文从
`out_dim * diag_dim`（QKV 和 FF 都是 48）降到 2 个。

`parallel_diagonal_pc_mult` 保留了（对着 `he.py` 读的时候有用，也有测试），只是 `pcmm` 不再用它。
`test_gpu_resource_fixes.py::test_pcmm_streams_the_grid_it_used_to_hold` 同时钉住两件事：流式版本和
网格版本结果逐位相同，以及 `pcmm` 确实不再建网格。

## 另一半 OOM：明文被反复编码，不是密文

你的日志里这一行值得注意：

```
Plaintexts loaded: 378 ~ 6426MB
```

6.4 GB 明文。原因是 `Stages.multiply` 把 numpy 掩码直接交给 `Engine.multiply`，而后者对
`np.ndarray` 分支是**每次调用现编一个满 tower 的 Plaintext**。掩码是固定的（`rotate_internal`
的两族、LayerNorm 的 value/statistic、FF 的窗口、pooler 的 CLS），一层里每个要用几十次。

已改：`Stages.plaintext()` 按**内容**缓存，引擎若提供 `encode_to_light_plaintext` 就转成 light
plaintext（level 无关，展开走引擎自己的缓存），`ClearEngine` 下是 no-op。一族掩码从此只编码一次。
这应该能把那 6.4 GB 砍到几十 MB，对 Bug 2 的显存压力也有帮助。

---

## Bug 5（`rotate(0)`）：你补的是症状，根因在 `Stages.rotate`

`Engine.rotate` 里加 `if delta == 0: return x` 是对的，我留着了。但**为什么会有人请求 index 0**才是
问题：`Stages.rotate` 把 THOR 的 delta 取负模 `slot_count`，delta 为 0（或 ±slot_count）时算出 0，
然后交给引擎——而 `ClearEngine.rotate` 会把每个被请求的 index 记进 `rotations_used`，所以
**`plan_rotation_keys` 的输出里带着 index 0**，GPU 侧就会去生成一把 OpenFHE 没有这个索引的密钥。

已在 `Stages.rotate` 里短路掉，`plan_rotation_keys` 的输出不再包含 0。顺带省掉一把 ~123 MB 的密钥
（按你日志里 49 把 6026 MB 算）。

---

## Bug 6（`plan_rotation_keys` 覆盖全层）：方向对，但那个 `try/except: pass` 必须去掉

改成跑完整 layer 是对的。但

```python
try:
    layer.forward(...)
except Exception:
    pass
return {k: v for k, v in ... if v >= 0}
```

这两行合起来会**静默返回一个不完整的 plan**——正是这个函数存在的意义所要防止的事情（GPU 上表现为
缺 rotation key）。而且它确实在掩盖一个真 bug：新版把几何写死成 THOR 的三个，`plan_rotation_keys(SMALL, ...)`
一进 `encode_weight_ff` 就 IndexError，被 `except` 吞掉后返回半个 plan，
`test_rotation_plan_covers_every_rotation` 因此失效。

已改：

* 加 `scope` 参数。`scope="layer"`（默认）跑完整层，需要生产几何；`scope="qkv"` 只跑 stage 01–05，
  任何几何都行——小几何的测试用这个。
* 去掉 `except`。dummy 用零是安全的：整层唯一依赖数值的控制流是 `he_inv` 的迭代次数，而那是标量决定的，
  所以零走的是和真权重完全一样的路径，抛出来的任何东西都是真 bug。
* 负 level 不再静默丢弃，改成带诊断的异常。

### 顺带发现：`bootstrap_level` 之前根本没传

`plan_rotation_keys` 建的是 `ClearEngine(g, depth=depth)`，用的是**默认 `bootstrap_level=14`**。
而 stage 12–14 光是 GELU 就要 13 个 level、整段要 17 个，所以自举之后层尾直接掉到 0 以下——
按你那版就是被 `v >= 0` 悄悄滤掉了 4 个 index。

已加 `bootstrap_level` 参数（默认 `depth - 10`）。实测：

| bootstrap_level | 结果 |
|---|---|
| 14 | 4 个 rotation 掉到 level 0 以下 |
| 30 | 1 个掉到 0 以下 |
| 80 | OK，**210 个 key**，最高 level 84 |

**这直接回答了 Bug 2 的关键前提**：自举之后至少要留够跑完 stage 12–16 的 level。

> **2026-09-08 更正**：上面这个「30 不够」的判据本身是不可靠的——它从**旋转的负 level** 反推饥饿，
> 而 plan 里每个 index 存的是**最大** level，所以只要两次旋转共用一个 index（`binary_rotations`
> 下全都共用），深处的负 level 就被浅处的大 level 掩掉了。已改成在 `ClearEngine` 里直接拦截任何
> 降到负 level 的操作。用这个可靠判据重测：**最小 bootstrap_level 是 38**，且与 depth 无关
> （depth=50/60/90 在 bl≥38 都通过，bl=36 都失败）。详见 `RESPONSE-gpu-oom-20260908.md`。

---

## Bug 2（bootstrap 运行时崩溃）：没 GPU，只能给方向

没法复现。但有两点可以先排掉：

1. 报告说崩在 `stage_02（make_rotated_copies）`——`stage_02_make_rotated_copies` 只有 `rotate`，
   没有 `multiply`。堆栈里的 `rotate_internal → multiply` 属于 **stage_03 的 `pcmm`**。
   定位到具体 stage 会好查很多。
2. 崩在 `EvalMultLightPt / EvalMultPt` 上，而上面那个「明文被反复编码」的问题正好在这条路径上，
   6.4 GB 明文 + 6 GB 密钥的情况下，`ExpandLightPlaintext` 的缓存行为值得先看一眼
   （`light_plaintext_cache` 默认 64 项，每项满 tower 在 depth=30/log_n=16 下约 10 MB）。

先把上面那些改动拉下来重跑，Bug 2 的现场可能会变。

---

## 本次改动清单

| 文件 | 改动 |
|---|---|
| `src/ConstantsGPU.cu` | `L + K <= MAXP` 检查（Bug 1 根因）**未编译** |
| `python/thorfhe/stages.py` | `rotate` 短路零旋转（Bug 5 根因）；`pcmm` 流式（Bug 3）；`plaintext()` 掩码缓存 |
| `python/thorfhe/he.py` | `plan_rotation_keys` 加 `scope` / `bootstrap_level`，去掉 `except`，负 level 报错（Bug 6） |
| `python/tests/test_gpu_resource_fixes.py` | 新增，7 个用例钉住上述三条 |
| `python/tests/test_stage6_thor_qkv.py` | 两处改用 `scope="qkv"` |

你补的 Bug 4（`add`/`subtract` 的 `np.ndarray` 分支）和 Bug 5 的引擎侧短路都保留，Bug 4 的
`level=self.depth - self.level(x)` 和 `multiply` 里的写法一致，没问题。

本机 79 passed / 40 skipped。
