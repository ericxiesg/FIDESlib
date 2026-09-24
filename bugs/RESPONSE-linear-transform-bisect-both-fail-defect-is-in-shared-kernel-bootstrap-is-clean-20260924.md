# LinearTransform bisect: 两个都失败 → 缺陷在共享 kernel;bootstrap 本身是干净的

2026-09-24,针对 `ed031a2`。

## 0. 二分结果:两个都失败

| 测试 | levelBudget | 路径 | 结果 |
|---|---|---|---|
| `LinearTransform` /8 (sparse) | {1,1} | 单步, stride=1, offset=0 | **FAIL** |
| `LinearTransform` /9 (thormod) | {1,1} | 单步, stride=1, offset=0 | **FAIL** |
| `CoeffsToSlots` /8 (sparse) | {3,3} | 多步 | FAIL (已知) |
| `CoeffsToSlots` /9 (thormod) | {3,3} | 多步 | FAIL (已知) |

**`ed031a2` §0 的二分结论:两个都失败 → 缺陷在共享的 `LinearTransform()` kernel / hoisted rotation 点积里,和多步无关。**

## 1. 关键数据

### 1.1 Bootstrap 输出(before LT)是干净的

| Config | CPU bits | GPU bits | 值 |
|---|---|---|---|
| sparse /8 | 47 | 47 | CPU=(0.25, 0.5, 0.75, 0.1, -0.1, ...) GPU=(0.25, 0.5, 0.75, 0.1, -0.1, ...) |
| thormod /9 | 38 | 38 | CPU=(0.25, 0.5, 0.75, 0.1, -0.1, ...) GPU=(0.25, 0.5, 0.75, 0.1, -0.1, ...) |

**Bootstrap (ModRaise + EvalMod) 的输出在 CPU 和 GPU 之间匹配**——前 8 个值完全一致,精度相同。这不是 28-bit 缺口的来源。

### 1.2 LinearTransform 输出(after LT)完全不同

| Config | CPU LT | GPU LT | Max Error |
|---|---|---|---|
| sparse /8 | (3.5e-15, **0.0450**, **-0.0615**, **0.0366**, **0.0673**, ...) | (5.7e-15, **0.0966**, **-0.0287**, **-0.0192**, **0.1123**, ...) | **0.0917** |
| thormod /9 | (1.4e-12, **0.0450**, **-0.0615**, **0.0366**, **0.0673**, ...) | (-2.5e-12, **0.0966**, **-0.0287**, **-0.0192**, **0.1123**, ...) | **0.0917** |

- CPU 和 GPU 的 LT 输出**结构性不同**——不是噪声,是完全不同的值。
- GPU LT 输出在两组参数间**完全相同**(除了第一个元素的噪声),因为输入和矩阵相同。
- CPU LT 输出也在两组参数间完全相同,同理。
- Max error 恒为 0.0917,与模数无关。

### 1.3 含义

**28-bit 缺口的来源是 LinearTransform kernel,不是 EvalMod 或 ModRaise。**

完整 bootstrap 的流程是:ModRaise → EvalMod → CtS (LinearTransform) → StC (LinearTransform)。

- ModRaise + EvalMod:CPU 和 GPU 匹配(47/47 bits for sparse, 38/38 for thormod)
- CtS + StC:GPU LinearTransform 产生完全错误的值
- 这导致完整 bootstrap 的 GPU 精度从 ~47 掉到 ~18(sparse),缺口 28 bit

## 2. GPU LT 输出的具体值(供对比)

```
CPU LT (两组相同, 除噪声):
(0.045006773, -0.061533577, 0.03659463, 0.067336695, 0.038927215, 0.04062699, -0.054047916,
 0.0027334531, -0.013629068, 0.010830834, -0.042172128, -0.10114594, -0.010772985, -0.045155508,
 0.053101962, ...)

GPU LT (两组相同, 除噪声):
(0.096602218, -0.028734409, -0.019231055, 0.11232964, 0.065735276, 0.12345397, -0.013820984,
-0.0038656864, 0.015170978, 0.030891853, 0.049536165, -0.12126512, -0.043431733, -0.09130121,
-0.017602557, ...)
```

这些不是"同一个值的噪声版本"——它们是完全不同的数。GPU LinearTransform 在计算一个不同的东西。

## 3. 下一步

缺陷在共享的 `LinearTransform()` kernel(`src/CKKS/CoeffsToSlots.cu:72` 和 `:198-201` 调用的同一个函数)。

需要查:
1. `LinearTransform()` kernel 本身——hoisted rotation 点积的实现
2. 对角线明文的编码——GPU 侧 `AddBootstrapPrecomputation` 生成的明文是否和 CPU 侧一致
3. 旋转索引——GPU 侧的 rotation key 索引是否正确

由于 bootstrap 输出是干净的(47/47 bits),问题不在 bootstrap 管线,而在 LinearTransform 这一步。
