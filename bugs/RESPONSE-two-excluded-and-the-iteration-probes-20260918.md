# 自举越界和 he_inv 空槽放大，两条都量过了，都不是

日期：2026-09-18。承接 `e265688`。

---

## 0. 你的更正是对的，和我这边一致

11a1/11a2 比值 GPU 203:1、clear 122:1，减法不构成相消。这条闭合了。

**关键不是 11a1 膨胀，是你数据里更早的一处**：

| 探针 | GPU | ClearEngine |
|---|---|---|
| 07c.denominator | 0.2925 | 0.3036 | ← **对的** |
| 07d.inverse_denominator | **5.3e+16** | 0.1475 | ← **错的** |

`07c` 是 `he_inv` 的输入，`07d` 是它的输出。**输入对、输出错 17 个数量级，
所以问题整个落在 `he_inv` 里面**，后面 10/11/11a1 全是它的下游。

还有一条贯穿你所有数据的特征：**p99 对得上，max 对不上。**

```
07a  p99 6.597 vs 6.526  ✓      max 37.73 vs 12.43  ✗
07d  p99 0.2381 vs 0.1092 ~     max 5.3e16 vs 0.1475 ✗
11a1 p99 4.349e7           ✗    max 4.76e7           ✗
```

是**少数 slot** 出问题，不是整体噪声。

---

## 1. 排除：自举越界（我原本最看好的一条）

我以为是消息超过 q0/(2Δ)，自举把它替换而不是刷新。**量了，不是。**

`clear.py` 现在记录每次自举的峰值，`bench` 打出来（真实 checkpoint，你那套 scale 参数）：

```
bootstrap headroom against q0/(2*Delta) = 16 (22 bootstraps)
              peak   of bound   headroom
            0.6537      4.1%     24.48x
             0.582      3.6%     27.49x
            0.5068      3.2%     31.57x
  tightest 24.48x
```

**最紧的一次也只用掉界的 4.1%，余量 24 倍。** 整条 schedule 离越界很远，
`--score-refresh-scale` / `--refresh-scale` / `--residual-scale` 都不需要再调大。

---

## 2. 排除：`he_inv` 放大空槽

`he_inv` 的迭代是 `a ← a·(2/k·δ_b − b)`。在 `b = 0` 的空槽里那个因子就是常数 `2/k·δ_b`，
所以 clear 上恰好为 0 的槽保持 0，而设备上 1e-12 的槽每轮翻倍。看起来很像。**量了，不够。**

真实几何、不同泄漏量，`he_inv` 出来的空槽最大值：

| 空槽泄漏 | 承载槽（应为 20） | 空槽 |
|---|---|---|
| 0（clear） | 20 | **0** |
| 1e-9 | 20 | 1.16e-05 |
| 1e-3 | 20 | **1** |
| 0.015 | 20 | **1** |
| 0.05 | 20 | **1** |

放大倍数就是 `2^迭代数`（11 轮 = 2048 倍），而且**到 1 就饱和**——因为 `a` 和 `b` 同样泄漏，
比值收敛到 1。空槽出来是 1 不是 0，确实是个瑕疵，但**做不出 5.3e16**。

---

## 3. 所以：`he_inv` 内部逐轮探针（已推）

`07c` 和 `07d` 之间隔着整个迭代，中间没有任何东西说话。现在 `THORFHE_DEBUG=1` 会打每一轮：

```
THORFHE_DEBUG=1 python -m thorfhe.bench fhe --engine fideslib ... --per-stage
```

**ClearEngine 基线（真实 checkpoint，第二次调用是 7 轮的那次）：**

```
07.inv_iter01_a  level 19  max +0.9807   p99 0.9795
07.inv_iter01_b  level 19  max +0.2139   p99 0.1204
07.inv_iter02_a  level 18  max +0.2286   p99 0.2272
07.inv_iter02_b  level 18  max +0.01636  p99 0.01608
07.inv_iter03_a  level 17  max +0.1377   p99 0.134
07.inv_iter03_b  level 17  max +0.003762 p99 0.003759
07.inv_iter04_a  level 16  max +0.1413   p99 0.1297
07.inv_iter04_b  level 16  max +0.003899 p99 0.003853
07.inv_iter05_a  level 15  max +0.1671   p99 0.1593
07.inv_iter05_b  level 15  max +0.003893 p99 0.003893
07.inv_iter06_a  level 14  max +0.1676   p99 0.1592
07.inv_iter07_a  level 13  max +0.1683   p99 0.1599
```

`a` 全程在 0.13–0.99，`b` 单调掉到 0.0039。**第一个对不上的 `iter` 就是答案。**

- 从 `iter01_a` 就错 → 问题在 `align(ones, bootstrap(denominator))`，即那次自举或 `ones` 本身
- 中间某轮开始发散 → 是 `_restore_magnitude` 的整数重缩放，或 `prepare_for_multiply`
- 一路正常到最后一轮 → 那就是 `he_inv` 之后、probe 之前的那一步

---

## 4. 顺带排除：factored key 的密钥计划是精确的

你问的 `07d` outlier 在 15 把密钥下不存在。我在本机验证了密钥计划本身：
跑一次干运行拿到计划，再用这套基跑第二次，记录每把密钥**实际被用在哪个 level**：

```
basis 21 keys, 3465 rotations, plan covers 21 keys
keys the run spent          21
keys the plan never built   0
keys used ABOVE their level 0
keys built but never spent  0
```

**没有缺的密钥，没有被用在超过自己 level 的密钥。** 所以 21 把这套本身是自洽的。
§3 的逐轮探针同时也会回答 outlier 的问题——如果 `iter01` 就错，那和旋转密钥无关。

（`§1` 的 §3 里我还是想要那两个对照跑：`--rotation-max-steps 2` 和 `--no-truncate-keys`，
各一次，能把"密钥"和"迭代"彻底分开。）

---

## 5. 现在的排除表

```
✅ 灾难性相消            （11a1/11a2 差 122 倍，条件数 1.008）
✅ 自举消息越界          （最紧余量 24.5 倍）
✅ he_inv 空槽放大       （饱和在 1，做不出 1e16）
✅ 旋转密钥计划          （21 把全覆盖，无超界使用）
✅ noise_level / 整数标量乘 / EvalNegate / 截断密钥
❓ he_inv 迭代内部       ← §3 的探针指向这里
```

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
