# 回复：你找到了，而且它在我写的代码里——补上让 ClearEngine 能抓到它

日期：2026-09-11。**C++ 本轮未改。**

## 一、这个 bug 是我的

`power_basis` 那行是我写的，而且它**违反了自己上面三行的 docstring**：

```python
def power_basis(self, x, powers):
    """``{k: x^k}`` for the requested exponents, each canonical."""   # <- "canonical"
    ...
    basis[k] = self.rescale(self.square(self.relinearize(basis[half])))   # <- 出来是 degree 2
```

`relinearize` 作用在**输入**上（而输入本来就是 degree 1，所以是空操作），`square` 的**输出**没人管。
你的根因分析完全正确，包括 `binomialMult` 悄悄丢掉 `c2` 那一步——那正是为什么结果是
「看着像数、其实错了」而不是崩溃。而且你指出其余四处平方都是对的顺序，这说明它就是个手误，
不是设计分歧。

## 二、真正要回答的问题：为什么 ClearEngine 没拦住

因为它**不建模 degree**。那个 no-op 上还写着理由：

```python
def relinearize(self, ct):
    """No-op: ciphertext degree is not modelled here, only the values, levels and scales."""
    return ct
```

这个项目一路是靠「把 ClearEngine 当成 FIXEDMANUAL 的类型检查器」推进的，
level 和 scale 都检查，**唯独 degree 不检查——而 degree 是这个契约的第三条**。
精确算术里没有 `c2` 可丢，所以它永远不会不高兴。这就是它为什么要靠几次 GPU 跑才被找出来。

补上了。`ClearCiphertext` 现在带 `degree`，并且按**设备实际的能力**来判定谁能接受 degree-2：

| 操作 | 设备带 `c2` 吗 | ClearEngine |
|---|---|---|
| `add` / `sub` | 带 | 允许，结果取较大的 degree |
| 乘明文 / 乘浮点标量 | 带 | 允许，degree 不变 |
| `rescale` / `level_down` / `copy` | 带 | 允许，degree 不变 |
| 密文 × 密文 | **不带**（`binomialMult` 丢 `c2`） | **拒绝** |
| 乘 i（单项式） | **不带**（`multMonomial`） | **拒绝** |
| 乘整数 | **不带**（`multIntScalar`） | **拒绝** |
| `rotate` / `conjugate` / `bootstrap` | 需要 degree 1（要 key switch） | **拒绝** |

报错长这样：

```
ScaleMismatch: multiply a ciphertext by a ciphertext: the ciphertext is degree 2;
relinearize it first. A lazy product has to be relinearised before anything that
key-switches it, multiplies it by another ciphertext, or multiplies it by a monomial
or an integer.
```

## 三、钉住

`tests/test_polynomial_reproducer.py` 加了两个用例，**都不碰设备**：

* `test_the_old_power_basis_order_is_now_caught`：照旧的顺序造一个 stale 的 degree-2 基元素，
  断言 `multiply` / `rotate` / `multiply_1j` **都抛**；
* `test_the_current_power_basis_order_is_canonical`：断言现在每个基元素 `degree == 1`，
  也就是 docstring 说的那句话。

**整套 96 个用例在打开 degree 检查后全过**，所以移植里没有别的 degree 违规——
这也是一次覆盖全项目的复查，不只是修一个点。

## 四、下一跑

`--per-stage --device-memory` 照旧。这次要看的：

* `07b.exp` 的量级——clear engine 的基准是 `max 2.28e-4`、`|x| med 1.35e-06`；
* `07c` / `07d` 跟着正常没有；
* `softmax` 那一行的 `scale` 和 `relRMSE`——期望回到 `scale ~1.0`、`relRMSE ~2.3e-06`。

上一轮那三处 `multScalar(-1.0)` 的修复仍然要编进去：`he_inv` 在 `07d`，
这次 `07b` 修好之后它才会轮到表态。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `python/thorfhe/clear.py` | `ClearCiphertext.degree`；按设备实际能力检查 degree-2 操作数 |
| `python/tests/test_polynomial_reproducer.py` | 两个用例钉住旧顺序会被拒、新顺序是 canonical |
