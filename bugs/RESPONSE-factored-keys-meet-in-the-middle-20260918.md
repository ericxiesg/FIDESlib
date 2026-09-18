# factored rotation key：真正省的不是挑哪几把，是一个 index 允许走几步

日期：2026-09-18。承接 `35841d9`。

---

## 0. 结论

同样 21 把密钥、同样 3.5 GiB，**每层旋转 4237 → 3465**。
差别只在 `RotationBasis` 允许一个 index 走几把密钥。

之前的实现：命中一把 → 1 步；两把之和 → 2 步；**其余全部退回二进制展开**。
grid 本身是二维的（`2032·j − 256·block`），所以 grid 点都是 2 步；但**贴着 grid 旁边**的那些
index 一个都不是，它们在付自己的 popcount——7 到 10 次旋转，而实际上离 grid 只差一把密钥。

`max_steps=4` 用 meet-in-the-middle 找 3~4 步的链。并且贪心**按运行时真正会用的分解来打分**，
所以它挑的是另外六把密钥：

```
max_steps=2   4237 rotations   extra=[2544, 10672, 26928, 30720, 32763, 32764]
max_steps=4   3465 rotations   extra=[2288,  6608, 10416, 14480, 30720, 32763]   <- 默认
max_steps=6   3461 rotations   （同上六把，多花 4 倍规划时间换 4 次旋转，不值）
```

代价：**+0.01 GiB**，规划多 10 秒。运行时零代价——分解是缓存的。

### 已在真实 checkpoint 上端到端验证

```
python -m thorfhe.bench fhe --engine clear --compact --layers 1 --limit 1 --per-stage \
    --refresh-after-dense --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
    --extra-rotation-keys 6

hidden after layer 0  MAE 1.441e-03  RMSE 1.847e-03  relRMSE 2.776e-03
```

和一 index 一把密钥的参考**逐位相同**。

---

## 1. 现在的取舍表（THOR geometry，depth 37，一层，`--compact`）

| 基 | 密钥 | 显存 | 旋转/层 |
|---|---|---|---|
| 一 index 一把 | 210 | 28 GiB | 1922 |
| binary | 15 | 2.5 GiB | 8258 |
| binary + 6 | 21 | **3.5 GiB** | **3465** |
| binary + 9 | 24 | 3.9 GiB | 3153 |

超过 9 把仍然塞不下。

---

## 2. 新增：按字节挑密钥

密钥是按 level 截断的，所以一把在 level 30 被用到的密钥是 level 3 那把的好几倍。
更反直觉的是——**加一把密钥可以让密钥集变小**：一个 level 30 的 index 退回二进制展开时，
会把它那一串 2 的幂全部顶到 level 30；给它自己的密钥，那些幂就只按别人要求的 level 留着。

给了 `--rotation-key-budget GIB` 之后，贪心按**每字节省下多少次旋转**排序，并且不越预算：

```
+6 不计价   21 keys  3465 rot  3.47 GiB
+6 计价     21 keys  3643 rot  3.38 GiB     <- 0.09 GiB 换 178 次旋转
+9 不计价   24 keys  3153 rot  3.93 GiB
+9 计价     24 keys  3199 rot  3.88 GiB
```

**只在卡装不下的时候用它**，默认不开。

---

## 3. 新的开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `--rotation-max-steps N` | 4 | 一个 index 允许经过几把密钥。2 = 旧行为 |
| `--rotation-key-budget GIB` | 无 | 密钥集的字节上限，同时切换到按字节挑 |
| `--special-primes K` | 11 | 只用来给密钥定价（`fhe` 之前没有这个，`budget` 有） |

`plan_rotations` 也补了 `demand` / `demand_levels`（干运行原始的 `{index: 次数}` 和
`{index: 最高 level}`），这样选基就不用再跑一遍干运行。

**另外修了一处：`bench fhe` 的 GPU 分支没有把 `compact` 传给 `plan_rotations`**，
只有 clear 分支传了。也就是说设备上带 `--compact --extra-rotation-keys` 时，基是按**不会发生的**
旋转频次挑的。现在两条路径一致。

---

## 4. 顺带发现：depth 37 的一层**必须**带 `--refresh-after-dense`

新探针 `10.attention_dense` 和 `11.residual`：

```
不带 --refresh-after-dense    10.attention_dense level 1    11.residual level 1   -> LN1 炸
带   --refresh-after-dense    10.attention_dense level 1    11.residual level 20  -> 通过
```

stage 10 出来就是 **level 1**，而 LN1 整条链要 14 层。`--refresh-after-dense` 不是可选优化，
是 depth 37 下唯一能跑通的路。它**不是默认值**，所以任何不带它的一层跑法都会在
`he_layernorm` 第 118 行 rescale 到 level -1。

同时，真实 checkpoint 必须带 `--residual-scale 256 --refresh-scale 4 --score-refresh-scale 16`，
否则 stage 15 的 bootstrap 会撞界（实测 93.51 vs q0/(2Δ)=16，584%）。
完整可跑的调用就是上面 §0 那条。

---

## 5. 请在设备上测

1. `--extra-rotation-keys 6`（现在默认 `--rotation-max-steps 4`）对比 `--rotation-max-steps 2`，
   报回每层 wall clock。预期旋转数 3465 vs 4237。
2. `--rotation-key-budget 3.4` 能否把密钥挤进去；报回 `AddRotationKeys` 期间的峰值。
3. 上面 §4 的两行探针 level，确认设备和 clear engine 一致。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
