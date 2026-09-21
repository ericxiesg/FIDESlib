# 逐槽探针数据：第一次 he_inv 正常，第二次从 iter01_b 就全槽发散

日期：2026-09-21。针对 `e853110`。**`THORFHE_DEBUG=1 --per-stage --inverse-lift 3`，热 cache，55.8s。**

---

## 0. 你要的三行

### 第一次 he_inv（正常收敛）

```
07.inv_iter01_b  level 19  min +0.023    max +1.529     p50 0.2733   p99 0.275      <0 0/32768     >1 1/32768
07.inv_iter02_b  level 18  min -0.9161   max +0.0485    p50 0.013    p99 0.02583    <0 2/32768     >1 0/32768
07.inv_iter03_b  level 17  min -0.04582  max +61.09     p50 0.003323 p99 0.003722   <0 2/32768     >1 1/32768
```

**`min` 从 iter01 就是正的**（+0.023）——PADDING_FLOOR 生效了。
`<0` 计数 0→2→2，`>1` 计数 1→0→1，**几乎全槽在 `[epsilon, 1]` 范围内**。
iter03_b 的 max=61.09 是个别槽（p99=0.003722），不影响收敛。

### 第二次 he_inv（从 iter01_b 就全槽发散）

```
07.inv_iter01_b  level 19  min -5.214e+06  max +7.196e+06   p50 3.327e+05   p99 2.918e+06   <0 16704/32768  >1 16064/32768
07.inv_iter02_b  level 18  min -4.883e+13  max +2.871e+13   p50 1.87e+11    p99 9.76e+12    <0 16320/32768  >1 16448/32768
07.inv_iter03_b  level 17  min -6.022e+28  max +7.924e+28   p50 2.152e+24   p99 5.673e+27   <0 16200/32768  >1 16568/32768
```

**iter01_b 就已经有 ~50% 槽 `<0`、~50% 槽 `>1`**，量级 1e+06。
iter02_b 平方爆炸到 1e+13，iter03_b 到 1e+28——标准 Goldschmidt 发散模式。

---

## 1. 两次 he_inv 的输入对比

两次 he_inv 的 `a`（correction）和 `b`（denominator）在 iter01 的探针：

| | 第一次（正常） | 第二次（发散） |
|---|---|---|
| iter01_a min | -4.935e-12 | -2252 |
| iter01_a max | +0.9977 | +2072 |
| iter01_a <0 | 12212/32768 | 16435/32768 |
| iter01_a >1 | 0/32768 | 4224/32768 |
| **iter01_b min** | **+0.023** | **-5.214e+06** |
| **iter01_b max** | **+1.529** | **+7.196e+06** |
| **iter01_b <0** | **0/32768** | **16704/32768** |
| **iter01_b >1** | **1/32768** | **16064/32768** |

**关键区别**：第一次的 `a`（correction）在 `[0, 1]` 范围内（max=0.9977），第二次的 `a` 已经到 ±2000。

**`a` 在进 he_inv 之前就已经坏了。** 这不是 he_inv 内部发散，是**输入**就发散了。

---

## 2. `a` 是什么

`he_inv` 的 Goldschmidt 迭代：
- `a` = correction（初始 = 1，每次迭代 `a = a * (2 - b*a)`）
- `b` = denominator（被求逆的数）

第一次 he_inv 的 `b` = softmax denominator（07c，max=0.3282，正常）。
第二次 he_inv 的 `b` = ？

**问题**：为什么第二次 he_inv 的 `a` 在 iter01 就到 ±2000？如果 `a` 初始 = 1，
那 `a_post_times = a * (2 - b*a)` 要到 2000，需要 `2 - b*a` 到 2000，即 `b*a` 到 -1998。
如果 `a=1`，那 `b` 要到 -1998。

**但 noise_level 全程 = 1**（124 条探针全 1），所以不是噪声问题。

---

## 3. 完整探针（07d 之前的所有迭代）

### 第一次 he_inv（5 次迭代，正常）

```
iter01_a  level 19  min -4.935e-12  max +0.9977     <0 12212  >1 0
iter01_b  level 19  min +0.023      max +1.529      <0 0       >1 1
iter02_a  level 18  min -2.272      max +0.2693     <0 12187  >1 0
iter02_b  level 18  min -0.9161     max +0.0485     <0 2       >1 0
iter03_a  level 17  min -36.05      max +0.06568    <0 12324  >1 0
iter03_b  level 17  min -0.04582    max +61.09      <0 2       >1 1
iter04_a  level 16  min -3.138e-08  max +6.958e+05  <0 12295  >1 1
iter04_b  level 16  min -1.179e+06  max +0.003901   <0 3       >1 0
→ 07d = iter04_a: max +6.958e+05  (1 slot >1)
```

### 第二次 he_inv（6 次迭代，从 iter01 发散）

```
iter01_a  level 19  min -2252       max +2072        <0 16435  >1 4224
iter01_b  level 19  min -5.214e+06  max +7.196e+06   <0 16704  >1 16064
iter02_a  level 18  min -1.167e+10  max +9.753e+09   <0 16332  >1 4176
iter02_b  level 18  min -4.883e+13  max +2.871e+13   <0 16320  >1 16448
iter03_a  level 17  min -7.54e+24   max +1.393e+25   <0 16706  >1 16062
iter03_b  level 17  min -6.022e+28  max +7.924e+28   <0 16200  >1 16568
iter04_a  level 16  min -5.849e+55  max +1.635e+56   <0 17029  >1 15739
iter04_b  level 16  min -3.202e+60  max +1.476e+44   <0 31924  >1 844
iter05_a  level 15  min -2.001e+117 max +1.281e+118  <0 16494  >1 16274
iter05_b  level 15  min -2.974e+123 max +1.875e+107  <0 24500  >1 8268
iter06_a  level 14  min -4.987e+214 max +5.411e+214  <0 16381  >1 16387
iter06_b  level 14  min -5.543e+214 max +5.137e+214  <0 16183  >1 16585
```

每次迭代量级平方增长：1e6 → 1e13 → 1e28 → 1e56 → 1e118 → 1e214。
到 iter06 已经超过 double 范围（1e308 附近）。

---

## 4. LayerNorm（11b）的探针

```
11a.variance     level 17  min -1.168e-07  max +8.542e+07   <0 15371  >1 2048
11b.inverse_sqrt level 10  min -2.11e+154  max +2.285e+154  <0 16264  >1 16504
```

variance 的 max=8.5e+07，但 `>1` 只有 2048/32768（padding 槽）。
inverse_sqrt 的 max=2.3e+154——和 07d 第二次一样的发散模式。

---

## 5. noise_level

全部 124 条 noise_level 探针 = 1。**没有漂移。**

---

## 6. per-stage fidelity

```
query    scale  2.0000   relRMSE 2.884e-07   ✅
value    scale  2.0000   relRMSE 3.141e-07   ✅
scores   scale  0.0313   relRMSE 2.874e-07   ✅
softmax  scale  5.4e+76   relRMSE 3.637e+02   ❌ (第二次 he_inv 发散)
```

query/value/scores 完全正确。从 softmax 开始爆。

---

## 7. timing

```
layer 0    53.02s  95.0%
total      55.80s
```

热 cache 55.8s（691 fields loaded, 0 encoded）。
比旧 install 的 90s 快了 ~40%——**GIL release + 2-tower 编码都生效了。**

---

## 8. 需要你看的

1. **第二次 he_inv 的 `a` 为什么在 iter01 就到 ±2000？** 第一次正常（a max=0.9977），
   第二次的 `a` 初始值是什么？是不是第二次 he_inv 的输入 `b`（denominator）本身就有问题？

2. **第二次 he_inv 是在哪调的？** softmax 里只有一次 `he_inv`，但探针打了两组。
   是 `he_exp` 内部也调了 `he_inv`？还是 `--inverse-lift 3` 导致了两次？

3. **PADDING_FLOOR 在第一次 he_inv 生效了**（min=+0.023，正数）。第二次的 min=-5.2e+06
   说明 floor 没帮上忙——因为 `a` 不是 padding 槽的问题，是**活跃槽**的 `a` 就坏了。

完整日志在 `/home/zhiyuan/bench-run/gpu_debug_hot.log`（219 行）。
