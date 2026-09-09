# 回复 GPU-bootstrap-works-level-budget-20260909

日期：2026-09-09。本机没有 GPU。

**第一次 GPU 自举跑通了**，精度 1.74e-5——对齐修复是对的，而且不伤精度。那行日志也把最后一环钉死了：

```
CtS layer 0 holds 34 limbs; a ciphertext at L=34 has 35 (scaling technique 1)
```

差 1，FIXEDMANUAL，和预测完全一致。

新的报错不是 bug，是**这次修复的真实代价没有被算进预算**。下面是重新算过的账。

---

## 一、那一层 level 是真的没了，两种改法都省不回来

对齐消耗一层，这层**拿不回来**：那一层根本没有对应的明文对角线可乘。

所以你的方案 2（变换之后再 grow 回来）不行——**密文的 level 是不可逆的**。grow 只是把 limb 数补上，
但那些新 limb 上没有正确的数据，长回来的是垃圾。

我也考虑过改 ModRaise 让它直接长到对角线那一层（`cc.L - 1`），这样就不用降。但那只是**把同一层挪个位置**：
起点低一层，自举后的 level 一样。所以两种改法净效果相同，这层确实是没了。

结论：**对齐是对的，代价是真的，正确的响应是把它算进预算，而不是想办法绕过。**

## 二、重新算过的数

三个数字，两个来自你的实测，一个来自我这边重测：

| | 值 | 来源 |
|---|---:|---|
| `GetBootstrapDepth` (3,3) | 16 | 你的日志 |
| 对齐额外消耗 | 1 | 本次修复 |
| **有效 bootstrap depth** | **17** | 16 + 1 |
| **一层需要的自举后 level** | **20** | 我重测（refresh + binary rotations 下，19 挂 20 过） |

所以 **最小 depth = 20 + 17 = 37**，不是 34。你跑的 depth=34 给出自举后 level 17 < 20，
正好在 stage 10 撞墙——完全对得上。

重新扫一遍：

| depth | 自举后 level | 显存 | 余量 |
|---:|---:|---:|---:|
| 34 | 17 ← 不够 | 27.4 GiB | +4.6 |
| **37** | **20 刚好** | **28.7 GiB** | **+3.3** |
| 38 | 21 | 29.1 GiB | +2.9 ← 低于 keygen 要的 3 GiB |
| 40 | 23 | 30.0 GiB | +2.0 |

**depth=37 是唯一同时满足两边的点**：level 墙刚好够，余量 3.3 GiB 刚好高过 keygen 的临时需求。
再往上显存余量就不够 keygen 了。窗口很窄，但存在。

## 三、改了三处

**① `budget.py` 记下实测的 bootstrap depth**（(3,3)=17、(4,4)=18），
`bench` 的 `--bootstrap-depth` 默认改成从这张表查，不再用我之前拍的 14。
没测量过的 level budget 会直接报错要求显式传，不再默默用一个猜的值——
之前那个 14 正是这次算错 depth 的原因。

**② `Engine.bootstrap(keep_levels=)` 少给就报错，不再静默。**

```python
surplus = self.level(out) - keep_levels
if surplus > 0: ...     # ← 少给的情况直接被忽略了
```

这就是为什么问题在 stage 10 才以 `EvalLevelReduce would drop every RNS limb` 冒出来——
**错误的操作、错误的位置**。现在自举当场就说：

```
bootstrap left level 9, but keep_levels=10 asked for more.
Raise the depth, or lower what the circuit expects after a bootstrap.
```

**③ `test_stage4_bootstrap` 改成测契约，不测常数。**

原来写死 `assert e.level(out) == 10`。但「自举留下几层」会随 level budget、密钥分布、
以及对角线编码 level 变化——写死一个数就是把测试绑在一组参数上。现在改成：先问它给多少，
然后验证 `keep_levels` 比它小时**精确兑现**、比它大时**报错而不是少给**。这样参数变了测试还在测真东西。

---

## 四、下一次运行

```bash
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 --per-stage \
    --binary-rotations --refresh-after-dense --depth 37 --dnum 4 \
    --bootstrap-level-budget 3,3 --light-plaintext-cache 8
```

`--bootstrap-depth` 不用传了，会自己查到 17。

跑通的话，这是**第一次在 GPU 上跑完 THOR 的一整层**。`--per-stage` 那张表的 **scale 列**要和
`docs/thor_port.md` 里 clear engine 的对上（query/value 2.0、scores 0.5、softmax 1.0023、
其余 2.0）；残差会比 clear engine 大，那是 CKKS 噪声，是预期的。**哪一级的 scale 先偏，就是哪一级有问题。**

顺带请发回来：单层耗时、峰值显存、以及 `key memory` 那一行。这三个数是 T5 剩下的全部内容。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/budget.py` | 记录实测 bootstrap depth（(3,3)=17 含对齐那层、(4,4)=18） |
| `python/thorfhe/bench.py` | `--bootstrap-depth` 默认从实测表查；没测过的 budget 直接报错 |
| `python/pyfideslib/__init__.py` | `bootstrap(keep_levels=)` 少给时抛错，不再静默返回更少的层 |
| `python/tests/test_stage4_bootstrap.py` | 改测契约（兑现 / 超出报错），不再写死具体 level |
