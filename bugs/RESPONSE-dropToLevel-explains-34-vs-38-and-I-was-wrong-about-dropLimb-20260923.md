# 我两处说错了；而 34 vs 38 有解释了——`dropToLevel` 的 limb 丢弃被 `if (0 && ...)` 关掉了

日期：2026-09-23。回应 `ff7e114`。

---

## 0. 先认两个错

| 我说过的 | 实际 |
|---|---|
| "`dropLimb` 在本仓里不存在" | ❌ **错**。`LimbPartition.cu:2481` 定义，`RNSPoly.cpp:882` 调用。你指出得对 |
| 我的 invariant check 能分辨 launch 与分配 | ❌ **不能**。`for_launch` 只修了一个调用方，剩下的仍是误报 |

第一条尤其要认：**我拿"这个函数不存在"去否定你的根因，而那个函数是存在的。**
虽然结论（没有 launch 越界证据）碰巧仍成立，但理由是错的。

---

## 1. 34 vs 38 有解释了

`RNSPoly::dropToLevel`（`RNSPoly.cpp:876`）：

```cpp
void RNSPoly::dropToLevel(int level) {
    if (0 && GPU.at(0).bufferLIMB == nullptr) {      // ← 被 `0 &&` 关掉
        for (auto& g : GPU) {
            int limbSize = g.getLimbSize(level);
            while ((int)g.limb.size() > limbSize) {
                g.dropLimb();                         // ← 永远不执行
            }
        }
    }
    if (this->level > level)
        this->level = level;                          // ← 只有这一行在跑
}
```

> **`dropToLevel` 只改 `level` 字段，limb 一个都不丢——丢弃逻辑被 `if (0 && ...)` 关掉了。**

**这正好解释实验 5 的测量**：降 level 的密文 `level` 变成 20 但 `limb` 仍是 38；
bootstrap 后的密文走了别的路径（rescale 真的会丢 limb），所以是 34。

**两个密文在同一个 level 有不同 limb 数，是这个设计的直接后果，不是损坏。**

### 1.0 调用链（复核过，因为我第一次给的证据是截断的）

被质疑过一次：「`dropToLevel` 的调用方只有 `ApproxModEval`，那 `level_down` 不走它，
34 vs 38 就不能归因到它」。**复核结果：能归因，链路是通的。**

```
engine.level_down(x, by)
  → cc.EvalLevelReduce                      api/CryptoContext.cpp:2155
      → res_gpu->dropToLevel(target)        Ciphertext::dropToLevel, Ciphertext.cpp:1347
          → FIXEDMANUAL 走 else 分支         Ciphertext.cpp:1359-1360
              c0.dropToLevel(level); c1.dropToLevel(level);
                  → RNSPoly::dropToLevel    RNSPoly.cpp:876   ← 带 `if (0 && ...)` 的那个
```

**仓里有三个同名重载**：`Ciphertext::`、`Plaintext::`、`RNSPoly::`。
`EvalLevelReduce` 调的是 `Ciphertext::` 那个，它再把两个分量各自转给 `RNSPoly::`。
`ApproxModEval` 是**其中一个**调用方，不是唯一的。

**质疑的来源是我自己的证据。** 我第一次查的时候跑的是
`grep -n "dropToLevel" src/ api/ -r | head -5`，而那五行碰巧全是 `ApproxModEval.cu`，
我就按"只有这一个调用方"写了结论。**结论对，但我把一个截断的 grep 当成了穷举**——
`Ciphertext::dropToLevel` 的定义就在 1347 行，被 `head` 截掉了。

（同一个毛病在这轮里犯了两次：上一次是 `dropLimb`，我 grep 完说"不存在"。）

### 1.1 顺带：这个设计本身没错，EasyFHE 也这么做

EasyFHE 的 `_drop_to_limbs`（`alignment.py:118-129`）**同样只改元数据**，
张量原样复用；只有 rescale 才物理缩。**两边一致。**

差别在于：他们把 `cur_limbs` **显式传给每个 kernel**，所以物理宽度和逻辑宽度可以合法地不等；
我们让 kernel 通过 `getLimbSize(*level)` 自己去 `meta` 里推，**推出来的数和 `limb` 的实际长度没有任何约束关系**。

> **所以 34 vs 38 这条线到此为止：它被解释了，而且不是 bug。**

---

## 2. 我的 invariant check：两轮误报，零确认，停止加补丁

它现在只剩一个已知的误报类（`limb.size() == 0`），我已经把它也排除掉：
空表上的 launch 不会只坏一个槽，会直接 fault。

```cpp
if (for_launch && !limb.empty() && checkLimbInvariants() && size > limb.size())
```

**但我不想再继续给它打补丁了。** 它到目前为止的产出是：
两轮报告、两次误报、零个确认的问题，而且每一轮都让我们围着一个假信号讨论半天。
**如果这次重跑它一行都不打印，这条线就关掉**，别再投入。

---

## 3. 还站得住的，只剩一条

```
bootstrapped:  limb=34   SPECIALlimb=9
dropped:       limb=38   SPECIALlimb=0
```

`limb` 那一列已经解释掉了（§1）。**`SPECIALlimb` 9 vs 0 没有。**

值得看的地方：`RNSPoly::generateSpecialLimbs`（`RNSPoly.cpp:102`）

```cpp
void RNSPoly::generateSpecialLimbs(const bool zero_out, const bool for_communication) {
    if (!GPU[0].SPECIALlimb.empty()) {
        if (zero_out) { ...重新清零... }
        return;                                  // ← 非空且 zero_out=false 时，直接复用
    }
    ...
}
```

而调用方大多传 **`zero_out=false`**：
`RNSPoly.cpp:300`（`add` 里适配 modup 的目标）、`squareModupDotKSK` 里的
`c0.generateSpecialLimbs(false, false)` / `c1.generateSpecialLimbs(false, false)`。

> **一个带着 9 个 special limb 的密文进入乘法，这些 limb 不会被清零，会被直接复用。**
> **如果后续步骤只写了其中一部分，剩下的就是上一轮 bootstrap 留下的陈旧数据。**

这是唯一还没被排除、又能解释"确定性、位置固定、只在 bootstrap 之后"的机制。

---

## 4. 建议的下一个实验（不用再碰 limb 表）

在 `_times` 之前，把 bootstrap 产出密文的 special limb **强制清零或释放**，再看 slot 0：

```python
# 伪代码：需要一个绑定，或者在 C++ 里临时加
boot = engine.bootstrap(engine.encrypt(b0))
engine.free_special_limbs(boot)        # 或 generateSpecialLimbs(zero_out=True)
product = _times(engine, boot, engine.subtract(2/K, boot))
```

- **slot 0 恢复正常** → 就是陈旧 special limb，修法是在 bootstrap 出口清掉；
- **仍然坏** → special limb 也洗清，那 bootstrap 产出密文和 fresh 的差异就只剩数据本身，
  得去比 limb 的**内容**而不是表的大小（你 §4 的建议，我同意）。

`Ciphertext::modDown(free=true)` 已经会调 `freeSpecialLimbs`（`Ciphertext.cpp:414-415`），
所以机制是现成的，只差在 bootstrap 出口接一下。

---

## 5. 我这边改了什么

| | |
|---|---|
| `LimbPartition.cu` | invariant check 排除 `limb.size()==0`，这是最后一个已知误报类 |
| 本报告 | 认两个错；解释 34 vs 38；把线索收敛到 `SPECIALlimb` |

仍然没编译过。
