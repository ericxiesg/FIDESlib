# 找到根因：rescale 后 noise_level 变负数，subPt 拒绝操作

日期：2026-09-18。针对 `6a31a4f`。

---

## 1. 根因

`subtract(ndarray, ct)` 在低 level 下**直接报错**：

```
FIDESlib: subPt with mismatched scales - the ciphertext is at noise level -11 and the plaintext at 1
```

**`rescale` 把 noise_level 变成了负数**（uint32 下溢）。

---

## 2. 逐 level 测试

在 GPU sb=59 下，encrypt 后连续 rescale，测 `subtract(ndarray, ct)`：

```
level 37  noise_level=1    subtract ✅  max_err=3e-13
level 25  noise_level=-11  subtract ❌  "mismatched scales"
level 17  noise_level=-19  subtract ❌
level 12  noise_level=-24  subtract ❌
level 11  noise_level=-25  subtract ❌  ← norm_1 的 level
level  9  noise_level=-27  subtract ❌
```

**noise_level 从 1 开始，每次 rescale 减 1，变成 0、-1、-2...**
uint32 下溢后显示为 4294967285（= 2^32 - 11）。

---

## 3. 为什么 norm_1 炸而 softmax 不炸

- **softmax**（stage 07）用 `subtract(scalar, ct)` → `EvalScalarSub`，
  不检查 noise_level 匹配 ✅
- **norm_1**（stage 11）的 `he_invsqrt` 用 `subtract(ndarray, ct)` → `EvalNegate(EvalSubPt(y, pt))`，
  `EvalSubPt` 检查 noise_level 匹配，密文 -11 vs 明文 1 → **拒绝/静默错误** ❌

在 ClearEngine 上 `subtract(ndarray, ct)` 是精确的（不检查 noise_level），
所以 ClearEngine 不炸。GPU 上 `EvalSubPt` 检查了，导致发散。

---

## 4. 为什么高 level 正常

level 37 时 noise_level=1（刚 encrypt），和 plaintext 的 noise_level=1 匹配。
但在 `norm_1` 里，密文经过 ~26 次 rescale（从 37 到 11），noise_level 变成 -25。

`he_invsqrt` 里 `self.subtract((3/k)*mask, a)` 的 `a` 是
`rescale(relinearize(multiply(...)))`，经过多次 rescale 后 noise_level 是负数。
而 `encode(mask, level=depth-level(y))` 生成的 plaintext noise_level=1。

---

## 5. 修法方向

### 选项 A：`EvalSubPt` 不检查 noise_level
最简单，但可能引入精度问题（scale 不匹配的加减法）。

### 选项 B：`encode` 时匹配密文的 noise_level
```python
pt = self.encode(x, level=self.depth - self.level(y))
# 改为：
pt = self.encode(x, level=self.depth - self.level(y), noise_level=self.noise_level(y))
```
需要 `encode` 支持 noise_level 参数，并且 GPU 的 light plaintext 能设 noise_level。

### 选项 C：`subtract(ndarray, ct)` 改用 `EvalSubPt(ct, pt)` + `EvalNegate`
当前代码已经是这样：
```python
return self.cc.EvalNegate(self.cc.EvalSubPt(y, pt))
```
问题在 `EvalSubPt(y, pt)` 检查 `y` 的 noise_level 和 `pt` 的不匹配。
**需要让 `pt` 的 noise_level 匹配 `y` 的。**

### 选项 D：rescale 不应该让 noise_level 变负
这可能是一个 FIDESlib 的 bug。FIXEDMANUAL 下 rescale 应该保持 noise_level >= 0。
需要检查 `Rescale` 的 GPU 实现。

---

## 6. 高 level 的 subtract + multiply 正常

```
level 37  subtract ✅ (3e-13)  multiply+rescale ✅ (1.3e-11)
```

在高 level 下 `subtract(ndarray, ct)` + 后续 multiply 完全正常。
**问题只在低 level（noise_level < 0）时出现。**

---

## 7. noise_level 是什么

在 OpenFHE/FIDESlib 的 CKKS 中，noise_level（也叫 NoiseLevel）表示
密文的"乘法深度"——encrypt 后是 1，每次 multiply 变 2，每次 rescale 减 1。
FIXEDMANUAL 模式下，rescale 不自动调整 scale，而是手动管理。

**noise_level 变负数意味着密文经过了比 encrypt 时更多的 rescale**——
这在正常使用中不应该发生，除非 bootstrap 后的 rescale 修复
（之前修的 NoiseLevel=2 bug）改变了 noise_level 的计数。
