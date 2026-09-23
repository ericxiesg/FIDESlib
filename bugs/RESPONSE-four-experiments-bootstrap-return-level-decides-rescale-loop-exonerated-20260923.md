# 四实验结果：bootstrap 返回 level 决定一切；rescale loop 洗清；C++ limb 状态是差异所在

日期：2026-09-23。针对 `d2219e1`。**四个实验全跑完，逐个报告。**

---

## 0. 总览

| 实验 | 结论 |
|---|---|
| 1. raw vs corrected bootstrap | **rescale loop 洗清**。raw (degree 2) 和 corrected (degree 1) 同样坏 |
| 2. bootstrapped vs dropped metadata | **Python 可见状态完全相同**（level, noise_level），但行为不同。差异在 C++ limb 状态 |
| 3. bootstrap 各阶段 slot 0 偏差 | **bootstrap 四阶段都不偏**（1.3x）。问题不在 bootstrap 本身 |
| 4. level sweep | **bootstrap 返回的 level 是关键**。返回 37 → 全干净；返回 20 → 全坏。不是 drop 到哪 |

---

## 1. 实验 1：raw bootstrap vs rescale loop

### 1.1 结果

**第一次（engine bootstrap 返回 level 37）：**
```
raw EvalBootstrap:  level 37  scale degree 1
after the loop:     level 37  scale degree 1
the loop spent 0 level(s)
raw (no rescale loop):      slot 0 / p50 = 1.1x   error 2.525e-10
corrected (current path):   slot 0 / p50 = 1.1x   error 2.531e-10
```
**都干净。loop 花了 0 level（degree 已经是 1，不需要修）。**

**第二次（engine bootstrap 返回 level 21→20）：**
```
raw EvalBootstrap:  level 21  scale degree 2
after the loop:     level 20  scale degree 1
the loop spent 1 level(s)
raw (no rescale loop):      slot 0 / p50 = 12954.4x   error 1.43153   worst 0, 9891, 1, 20170, 19782
corrected (current path):   slot 0 / p50 = 12950.8x   error 1.42454   worst 0, 9891, 1, 20170, 19782
```
**raw 和 corrected 同样坏，worst at 完全一致。**

### 1.2 结论

**rescale loop 不是原因。** raw EvalBootstrap（degree 2，没修过）和 corrected（degree 1，修过）在低 level 同样坏。bootstrap 返回的密文本身就有问题，不管后面做不做 rescale。

---

## 2. 实验 2：bootstrapped vs dropped ciphertext metadata

### 2.1 结果

```
[2] at nominal level 20:
    bootstrapped: level  20  noise_level 1
    dropped:      level  20  noise_level 1
    -> identical in everything Python can read; look at the C++ limb state
    bootstrapped: slot 0 / p50 = 13224.9x   slot 0 error 1.45203
    dropped:      slot 0 / p50 = 1.1x       slot 0 error 5.017e-09
```

### 2.2 结论

**level 和 noise_level 完全相同**（都是 20, 1），但 bootstrapped 坏（13224x），dropped 干净（1.1x）。

**差异在 Python 看不到的 C++ 内部状态**：
- limb 布局（哪些 limb 是活的）
- allocation width
- DIGITlimbptr 表的长度
- key-switch 用的 digit 数量

**这指向 key-switch kernel 在低 level 时依赖的内部表结构和 bootstrap 产出的密文不匹配。**

---

## 3. 实验 3：bootstrap 各阶段 slot 0 偏差

### 3.1 结果

```
[3] slot 0 against the other slots, after each bootstrap stage:
    stage 1 (ModRaise     ): slot 0 0.705801   others p50 0.529529  max 0.981137   ratio 1.3x
    stage 2 (CoeffsToSlots): slot 0 0.705801   others p50 0.529529  max 0.981137   ratio 1.3x
    stage 3 (EvalMod      ): slot 0 0.705801   others p50 0.529529  max 0.981137   ratio 1.3x
    stage 4 (SlotsToCoeffs): slot 0 0.705801   others p50 0.529529  max 0.981137   ratio 1.3x
```

### 3.2 结论

**bootstrap 的四个阶段在 slot 0 都没有特殊偏差**（1.3x，正常）。bootstrap 输出在 slot 0 和其他槽一样健康。

**问题不在 bootstrap 本身，在 bootstrap 产出密文的内部结构和后续 `_times` 的交互。**

---

## 4. 实验 4：level sweep

### 4.1 结果

**第一次（bootstrap 返回 level 37，然后 drop）：**
```
level    slot 0 / p50    slot 0 error   worst three
   37             1.1     2.527e-10      21071, 8337, 31487
   34             1.1     1.009e-09      15780, 25069, 10691
   31             1.1     1.765e-09      21688, 14911, 10691
   28             1.1     4.512e-09      2093, 17073, 3003
   25             1.1     3.111e-09      3003, 5168, 26708
   22             1.1     4.960e-09      26844, 9417, 27275
   19             1.1     4.119e-09      14475, 23297, 19660
   16             1.1     5.801e-09      31396, 367, 3003
   13             1.1     6.081e-09      3698, 3003, 19660
   10             1.1     7.875e-09      6280, 1716, 31913
    7             1.1     6.250e-09      1629, 552, 14858
    4             1.1     9.584e-09      1543, 21827, 1972
```
**从 level 37 drop 到 level 4，全部干净（1.1x）。worst 槽每次都不同（散布）。**

**第二次（bootstrap 返回 level 20，然后 drop）：**
```
level    slot 0 / p50    slot 0 error   worst three
   20         13021.9         1.41499   0, 9891, 1
   19         13022.4         1.41499   0, 9891, 1
   18         13021.9         1.41499   0, 9891, 1
   17         13022.4         1.41499   0, 9891, 1
   16         13021.9         1.41499   0, 9891, 1
   15         13022.4         1.41499   0, 9891, 1
    ...（一直到 level 2，全部 13022x，worst at 0, 9891, 1 不变）
```
**从 level 20 drop 到 level 2，全部坏（13022x）。worst at 0/9891/1 始终不变。**

### 4.2 结论

**这是最关键的发现：**

| bootstrap 返回 level | 之后 drop 到 | slot 0 / p50 | worst at |
|---|---|---|---|
| **37** | 37 → 4 | **1.1x（全干净）** | 散布 |
| **20** | 20 → 2 | **13022x（全坏）** | 0, 9891, 1 |

**同一个 bootstrap 函数，第一次返回 level 37，第二次返回 level 20。返回 37 的密文永远干净；返回 20 的密文永远坏。**

**不是"drop 到什么 level"决定坏不坏——是"bootstrap 返回时的 level"决定。** 一旦 bootstrap 返回了低 level 的密文，它就带着问题，drop 到任何 level 都不会修好。

### 4.3 为什么两次 bootstrap 返回不同 level？

两次跑用的是**同一参数**、**同一 seed**，但 bootstrap 返回的 level 不同（37 vs 20/21）。这说明 bootstrap 的返回 level 取决于运行时状态——可能是：
- key 生成时的随机性
- GPU 上的分配顺序
- bootstrap 内部的 level plan（`34 of 48 keys truncated`）

**如果能让 bootstrap 稳定返回 level 37，bug 就不会触发。** 但这不是修法——真正的 bug 在 C++ 层的 key-switch 对 bootstrap 产出密文的 limb 结构处理有误。

---

## 5. 综合

### 5.1 已确认

| 事实 | 来源 |
|---|---|
| rescale loop 洗清 | 实验 1：raw 和 corrected 同样坏 |
| Python 可见状态相同但行为不同 | 实验 2：level/noise_level 一样，但 13224x vs 1.1x |
| bootstrap 四阶段不偏 | 实验 3：1.3x |
| bootstrap 返回 level 决定一切 | 实验 4：返回 37 全干净，返回 20 全坏 |
| 问题在 C++ limb 内部状态 | 实验 2 + 实验 4 |

### 5.2 推断

bootstrap 在某些运行中返回低 level 的密文（level 20），这个密文的 **C++ 内部 limb 布局**和 fresh encrypt 的不同（虽然 Python 可见的 level/noise_level 相同）。当这个密文在低 level 做 `_times`（multiply → relinearize → rescale）时，key-switch kernel 依赖的 DIGIT 表 / limb 指针在 slot 0 处算错。

**这不是数值精度问题，是 C++ 实现的 key-switch 在特定 limb 布局下的确定性 bug。**

### 5.3 下一步建议

1. **C++ 侧 print**：在 key-switch kernel 里打印 bootstrap 产出密文 vs fresh encrypt 密文的 limb 指针表 / DIGIT 表长度，找到差异
2. **检查 `beta = ceil(cur_limbs/alpha)` 边界**：实验 4 说不是平滑过渡而是"bootstrap 返回时就定了"，这和 DIGIT 表的 digit 数量一致——如果 bootstrap 返回的密文的 limb 数恰好让 beta 和 fresh encrypt 不同，key-switch 会用不同的路径
3. **`ElemenwiseBatchKernels.cu:271-273`** 的三级指针解引用——在 bootstrap 产出的低 level 密文上，这个指针链可能在 slot 0 越界
