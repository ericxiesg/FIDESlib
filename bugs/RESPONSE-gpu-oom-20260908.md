# 回复 GPU-OOM-expand-light-pt-20260907

日期：2026-09-08。本机没有 GPU；显存数字来自一个**用你三轮日志标定过**的模型，代码改动跑过本机测试。

结论先行：**binary rotations 生效了**（210→15 把 key，51 GiB→3.6 GiB，截断也终于开始起作用），
但 level 预算和显存预算目前**不相交**——差 2.4 GiB。下面是算出来的账、一个能提前算账的工具、
以及一处我自己之前判据出错的更正。

---

## 一、先更正我自己的一个错误结论

上一轮我写过「实测 `bootstrap_level=30` 不够跑完一层」。**那个判据是错的。**

它从 `plan_rotation_keys` 里**旋转的负 level** 反推 level 饥饿，而 plan 对每个 index 存的是
**最大** level。只要两次旋转共用一个 index，深处的负 level 就被浅处的大 level 掩掉——
`binary_rotations` 下 15 把 key 全都被反复共用，所以这个判据直接失效了（表现为「bl=16 也通过」，
显然不可能）。

已改成在 `ClearEngine` 里直接拦截：任何 `rescale` / `level_down` 把密文降到负 level 就抛异常。
这是唯一可靠的位置。用它重测：

| | |
|---|---|
| **最小 bootstrap_level** | **38** |
| 与 depth 无关 | depth=50 / 60 / 90 在 bl≥38 全通过，bl=36 全失败 |
| 卡在哪 | `stage_13_gelu` 自举**之前**那次 rescale——即 softmax 自举到 GELU 之间要 38 个 level |

---

## 二、显存的账

标定：`Rotation keys loaded: 49 ~ 12152MB` → 248 MiB/key；按
`2 * dnum * (L+1+K) * N * 8` 反解出 **K = 11**。模型复现单 key 大小到 248.0 MiB，与日志完全一致。

`depth=50, dnum=4, 15 把 binary key, levelBudget (3,3)`：

| | |
|---|---:|
| rotation keys (15，截断) | 3.6 GiB |
| bootstrap keys | 11.6 GiB |
| bootstrap plaintexts | 10.2 GiB |
| 其它（工作缓冲、aux 池、eval/conj key） | 9.0 GiB ← **标定值，不是推导值** |
| **合计** | **34.4 GiB** |
| 32 GiB 卡 | **超 2.4 GiB** |

那 9 GiB 是从你 round 3 反推的：日志报 keys 12044 MiB + plaintexts 10395 MiB = 22.0 GiB，
而此时一次 ~27 MiB 的 light plaintext 展开已经分不出来了。**这是标定，我在代码里也是这么写的**——
不把它算进去，模型就会给出乐观的假结论，代价是再烧一次二十分钟的运行。

按 depth 扫一遍：

| depth | 合计 | 余量 |
|---|---:|---:|
| 44 | 31.8 GiB | +0.2 |
| 46 | 32.6 GiB | −0.6 |
| 48 | 33.5 GiB | −1.5 |
| 50 | 34.4 GiB | −2.4 |

**这就是矛盾所在**：bl≥38 意味着 depth 至少要 38 + bootstrap 自身深度（(3,3) 大约 14）≈ 50，
而 depth=50 超 2.4 GiB。

---

## 三、新工具：先算再跑

```bash
python -m thorfhe.bench budget --depth 50 --dnum 4 --keys 15
python -m thorfhe.bench budget --depth 44 --binary-rotations   # 自己跑 plan，按 level 逐 key 截断
```

`fhe` 子命令现在也会在建 context **之前**把这张表打到 stderr，超了立刻能看见。

对**没测量过**的 levelBudget，它会明确拒绝给总数：

```
bootstrap plaintexts     UNKNOWN - this level budget has never been measured.
                         Run it once and add the reported plaintext count and MiB to
                         budget.MEASURED_BOOTSTRAP. No total without it.
```

宁可说不知道，也不给一个看着能装下的假数。

---

## 四、`bench` 现在能直接驱动 GPU 了

之前 `make_engine` **根本没传 `bootstrap_level_budget`**，`rotation_indexes` 也只认命令行字符串——
所以你一直得手写复现脚本。现在：

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 --per-stage \
    --binary-rotations --depth 50 --dnum 4 --bootstrap-level 40 \
    --bootstrap-level-budget 3,3 --secret-key-dist sparse \
    --light-plaintext-cache 8 --card-gib 32
```

rotation plan 由 `plan_rotation_keys` 现算，不用手填。`--light-plaintext-cache` 默认从 64 改成 **8**：
每项是 `(depth+1) * N * 8` ≈ 27 MiB，64 项就是 1.7 GiB，这是白占的。

---

## 五、把 2.4 GiB 补回来，按性价比

1. **`--light-plaintext-cache 8`**（已是默认）：约 1.5 GiB。单这一条就把 depth=48 拉进来了。
2. **`levelBudget={4,4}`**：bootstrap 明文 10.2 GiB 是第二大头，(4,4) 会减少线性变换的 giant-step
   数，明文和 key 一起降。**没人测过**，所以工具拒绝估。跑一次把
   `Plaintexts loaded: <n> ~ <m>MB` 和 key 数发回来，我把它加进 `MEASURED_BOOTSTRAP`，之后就能算。
   代价是 bootstrap 深度增加，可能又把 depth 顶上去——所以要实测才知道净收益。
3. **精确量一下那 9 GiB**：在 engine 构造的各阶段之间打 `nvidia-smi --query-gpu=memory.used`
   （keygen 前 / rotation key 后 / bootstrap setup 后 / bootstrap keygen 后）。现在它是一个标定的
   黑盒；拆开之后大概率有能省的。

如果 1 就够，`depth=48, bl=38` 应该能跑起来——**这是我建议的下一次运行**。

---

## 六、关于你那四条建议

1. **减小 `light_plaintext_cache`** —— 采纳，默认已改 8，并加了命令行开关。
2. **`levelBudget={4,4}`** —— 同意，但需要先测一次才能算净收益，见上面第 2 条。
3. **bootstrap plaintext 按需加载** —— 方向对，但工作量大；先做 1 和 2。
4. **降 depth** —— 有上限：**bl≥38**，所以 depth 压不到 44 以下。这是硬约束，不是调参。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/clear.py` | 任何降到负 level 的 `rescale`/`level_down` 直接抛异常（可靠的饥饿判据） |
| `python/thorfhe/budget.py` | 新增。显存模型，按日志标定，未测量的配置拒绝给总数 |
| `python/thorfhe/bench.py` | `budget` 子命令；`fhe` 建 context 前打印预测；GPU 参数全部接上命令行 |
| `bugs/RESPONSE-gpu-benchmark-20260907.md` | 更正「bl=30 不够」那段 |
