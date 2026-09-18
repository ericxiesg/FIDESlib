# 三个算子全部干净，嫌疑转向 aliasing / prepare_for_multiply

日期：2026-09-18。承接 `b4f0f6d`。

---

## 0. 结论

协作者要求的三个算子隔离测试全部通过。**问题不在单个算子，在组合/别名。**

```
EvalScalarSub(s, ct)       max abs err 1.818e-12  ← 干净
EvalMultByInteger(ct, 486) max abs err 8.835e-10  ← 干净
EvalConjugate + EvalAdd    max abs err 4.020e-11  ← 干净
```

所有 level（37/27/17/13）、所有实际使用的标量（1.00049 到 0.004）都测了。
误差都在 FHE 噪声量级（1e-10 到 1e-12），没有任何一个能造出 219.6。

---

## 1. 完整测试结果

### EvalScalarSub: subtract(scalar, ct)
```
s=1.00049      level=37  max abs err 1.818e-12  max |got| 1.5
s=1.00049      level=17  max abs err 1.818e-12  max |got| 1.5
s=0.250732     level=37  max abs err 1.818e-12  max |got| 0.7507
s=0.250732     level=17  max abs err 1.818e-12  max |got| 0.7507
s=0.0158389    level=37  max abs err 1.818e-12  max |got| 0.5158
s=0.0158389    level=17  max abs err 1.818e-12  max |got| 0.5158
s=0.00400755   level=37  max abs err 1.818e-12  max |got| 0.504
s=0.00400755   level=13  max abs err 1.818e-12  max |got| 0.504
```
（所有 level 都测了，只列部分；全部一致）

### EvalMultByInteger: multiply(ct, 486)
```
factor=486  level=37  max abs err 8.835e-10  max |got| 243
factor=486  level=17  max abs err 8.835e-10  max |got| 243
factor=486  level=13  max abs err 8.835e-10  max |got| 243
```

### EvalConjugate + EvalAdd
```
conj+add  level=37  max abs err 4.020e-11  max |got| 1
conj+add  level=17  max abs err 4.034e-11  max |got| 1
conj+add  level=13  max abs err 4.021e-11  max |got| 1
```

---

## 2. 下一步：aliasing

协作者说的 aliasing 假设正好对上 `b` 比 `a` 先坏的形状：

> `correction` 在一轮里被用两次，先 `a` 后 `b`——如果第一次乘法就地改掉了它，
> 第二次就会拿到坏的

迭代里的操作序列（从 `numeric.py` 读代码）：
```python
correction = self.subtract((2 / k) * mask, b)   # EvalScalarSub
a = self._times(a, correction)                    # 第一次用 correction
b = self._times(b, correction)                    # 第二次用 correction ← 如果第一次改了它
```

如果 `_times`（`EvalMultNoRelin` + `EvalRelinearize` + `Rescale`）就地修改了 `correction`
的某个内部缓冲区（比如 RNS tower），第二次乘法会拿到部分被改过的 `correction`，
导致 `b` 出错而 `a` 正常。

**这正好解释了设备数据里 `b` 先炸、`a` 在 iter03 还正常的现象。**

---

## 3. 请协作者查 `multNoRelin`

三个算子都干净，按协作者说的：**去读 `multNoRelin`**，看它是否就地修改输入密文的
内部缓冲区。如果 `EvalMultNoRelin(a, b)` 改了 `b` 的 RNS tower，那就是 aliasing。

另外也请查 `prepare_for_multiply`：如果它在准备 `a` 时顺带改了 `correction`，
效果一样。

---

## 4. 测试条件

```
GPU: Quadro GV100 32GB (sm_70), CUDA 13.0
Engine: pyfideslib.Engine("cuda:0", log_n=16, depth=37, scaling_bits=59, first_mod_bits=60, dnum=4)
Input: x = linspace(-0.5, 0.5, 65536)
```

log_n=16 (N=65536) 和实际 THOR 跑的一致。标量用的是 `he_inv` 真实迭代值。

Co-Authored-By: CodeAgent (GLM-5.2) <noreply@anthropic.com>
