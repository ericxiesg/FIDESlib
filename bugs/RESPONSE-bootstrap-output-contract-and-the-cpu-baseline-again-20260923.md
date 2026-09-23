# Bootstrap 输出契约（仿 EasyFHE 的 `BootstrapProgram`）；另：CPU 参照**又**失效了

日期：2026-09-23。回应 `db84cca`。

---

## 0. 两件事

| | |
|---|---|
| **GPU stage 2 重复跑不一致（78.8% 槽 CV>1）** | ✅ **GPU 内部比较，成立**，而且是本轮最重要的发现 |
| 「CPU 完全稳定」作为对照 | ❌ **又失效了**——你们没重编译，CPU 仍在跑完整 bootstrap（§1） |
| 但 stage-2 不确定性是真 bug 还是观测假象 | ❓ **未分辨**，判别实验见 §2 |
| EasyFHE 式输出契约 | ✅ **已实现**（§3） |

---

## 1. CPU 那一列还是无效的，而且有直接证据

你报的 CPU slot 0 = **0.705801**，六次一致。

**这个数就是「完整 bootstrap」的值。** 证据是你自己早先那份增量构建的日志：

```
stage 1 (ModRaise     ): slot 0 0.705801
stage 2 (CoeffsToSlots): slot 0 0.705801     ← 四个 stage 全是它
stage 3 (EvalMod      ): slot 0 0.705801
stage 4 (SlotsToCoeffs): slot 0 0.705801
```

那次四个 stage 相同，正是因为 `stopAfterStage` 被丢掉、**每次都跑完整 bootstrap**。
**所以 0.705801 = 完整 bootstrap 的 slot 0。**

而你现在的 CPU「stage 2」也是 0.705801 → **CPU 跑的仍然是完整 bootstrap。**

我在 `e339ff3` 已经让 CPU 分支在 `stopAfterStage != -1` 时抛异常——
**如果那个改动在你的构建里，CPU 这一列会抛错而不是给出数字。** 所以你们还没重编译。

> **CPU 稳定、GPU 不稳定，比的仍然是「完整刷新」对「中间态」。这条对照要作废第二次。**

---

## 2. 但 GPU 那一列成立——只是还不知道它说明什么

**同一个输入、同一个 engine、跑六次，GPU stage 2 给出不同结果**——
这是纯 GPU 内部的比较，不需要 CPU，成立。

**但有两种解释，目前分辨不了：**

| | 含义 |
|---|---|
| **A. CtS 真的有竞态** | 少了流同步/事件，结果取决于哪个 kernel 先完成。**那就是 bug 本身。** |
| **B. 观测假象** | `stopAfterStage` 提前 `return`，**跳过了后面所有步骤**。中间态的 scale 是飞行中的，而且提前返回可能绕过了正常路径才会做的同步。读一个还没落定的状态，本来就会每次不同。 |

**B 是我的仪器的问题，而这一轮我已经栽过两次了**（`for_launch` 误报、CPU 分支静默忽略）。
所以在排除 B 之前，我不认为 A 成立。

### 2.1 判别实验（一条命令，不用改代码）

**把完整 bootstrap 跑六次**（`stopAfterStage = -1`，也就是普通的 `engine.bootstrap`），
同样逐槽比对：

- **完整 bootstrap 也不确定** → **A 成立，是真竞态**，而且直接就是我们找的 bug；
- **完整 bootstrap 逐位一致** → **B 成立**，stage-2 的不稳定是读中间态的假象，
  这条线关掉，回到「为什么乘法坏 slot 0」。

**这个实验比继续挖 stage 2 重要得多**，因为它决定前面那 78.8% 是不是线索。

---

## 3. 输出契约：已实现

仿 EasyFHE 的 `BootstrapProgram`（`spec.py:97-106`）：生成期钉死输出状态、
入口校验 context fingerprint、出口归一。我们的版本放在
`BootstrapPrecomputation` 里——它正是我们这边「生成期产物、按 slots 查表复用」的等价物。

### 3.1 契约本身

```cpp
struct OutputContract {
    int noiseLevel = 1;      // 声明的：FIXEDMANUAL 下的规范 scale degree
    int level      = -1;     // 第一次 bootstrap 钉住，之后逐次校验
    bool levelPinned = false;
};
struct Fingerprint { int logN, L, dnum, K, rescaleTechnique; };
```

### 3.2 在 bootstrap 出口执行（`Bootstrap.cu`，仅 `stopAfterStage == -1`）

1. **fingerprint**：首次使用时记录，之后不符就抛——
   预计算的对角线是按当时的模数链编码的，换了 context 再用会表现成「数值误差」，
   而不是「配置错了」。
2. **`noiseLevel` 是声明值，不是观测值**：FIXEDMANUAL 就该是 1，
   `approxModReduction` 本来就为此 rescale 一次。没做到就在**这里**补，
   而不是在一个不知道自己花了几个 level 的调用方里补。
3. **`level` 首次钉住、之后强制一致**：预测它要把 level budget、secret key dist、
   对角线编码 level 三样都复算一遍；观测一次再守住，同样能抓漂移。

### 3.3 这解决了什么

`72dc818` 那次：`EvalBootstrap` 在 FIXEDMANUAL 下返回 degree 2，
而 degree 只在乘法时涨、rescale 时降，所以从 2 起步到 softmax 结束就是 12108，
**第一个 addPt 才因为 scale 不匹配失败**——在那之前一路静默。
当时的修法是在 Python wrapper 里加 rescale 循环：**离出问题的算子很远，
而且悄悄吃掉了一个 level**（`depth 37, bootstrap→19` 直接 FAIL 就是它）。

现在：**违约在 bootstrap 自己的出口被抓到并修正**，而且第一次发生时会打印
「rescaled N time(s)，每次花一个 level」，那个代价不再是隐形的。

### 3.4 Python 侧的 rescale 循环怎么办

**先不动。** 它的条件是 `while noise_level > 1`，C++ 做完之后它看到 1 就直接跳出——
自动变成 no-op。**等设备验证 C++ 这段确实生效，再把它删掉。**
（现在删，如果 C++ 这段有问题，就同时失去了兜底和诊断。）

---

## 4. 我这边改了什么

| | |
|---|---|
| `BootstrapPrecomputation.cuh` | `OutputContract` + `Fingerprint` |
| `Bootstrap.cu` | 出口执行契约；仅完整 bootstrap，中间态跳过 |

**没编译过**（本机无编译器）。**而且请先跑 §2.1**——那个比契约更急。
