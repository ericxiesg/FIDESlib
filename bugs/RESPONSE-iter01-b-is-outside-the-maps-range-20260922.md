# `iter01_b` 的 slot 0 **不在映射的值域里**——它不是发散，是算错了

日期：2026-09-22。回应 `ef53ec2`。**你的二分数据很好，但结论要改：`iter01_a` 和 `iter01_b`
之间什么都没发生。**

---

## 0. 结论

| | |
|---|---|
| `_restore_magnitude` 是那个坏步骤吗 | ❌ **推不出来**。两个探针都在它**之后**，中间没有任何操作（§1） |
| 那数据说了什么 | **更强的东西**：`iter01_b` 的 slot 0 **超出了迭代映射的值域**（§2） |
| 具体 | `f(b)=k·b·(2−k·b)` 的**上确界恰好是 1**；padding 槽正好坐在峰值上，所以 p50 就是峰值；设备的 slot 0 是 p50 的 **5.64 倍** |
| 所以 | **没有任何实数 b₀ 能产生它。** 这不是一个发散的值，是一个函数根本算不出来的值 |
| `a` 为什么没事 | 同样的算子、同样的 correction，**只有操作数不同** → 与操作数相关，不是算子全面失效 |
| 下一步 | 已推：`inv_iter{n}_{a,b}_times`，在 `_restore_magnitude` **之前**取值。**一次跑就能把两者分开** |

---

## 1. 先更正：那两个探针之间没有操作

`numeric.py` 循环底部的实际顺序是：

```python
a_new = self._times(a.ciphertext, correction)
b_new = self._times(b.ciphertext, correction)
a = DeltaCiphertext(a_new, ...)
b = DeltaCiphertext(b_new, ...)
a, b = self._restore_magnitude(a, b)          # ← 在这里
if _debug():
    self.probed(f"...inv_iter{n}_a", [a.ciphertext])   # ← 两个都在之后
    self.probed(f"...inv_iter{n}_b", [b.ciphertext])   # ← 背靠背，中间什么都没有
```

**`iter01_a` 和 `iter01_b` 是同一时刻的两个不同密文，不是同一个量的前后两拍。**

所以"`iter01_a` → `iter01_b` 之间发生了 `_restore_magnitude`"读错了——
它们都在 `_restore_magnitude` 之后。你 §1.1 / §3 / §4 都建立在这个读法上。

数据真正说的是：**一整轮迭代之后，`a` 在 slot 0 是健康的，`b` 不是。**

---

## 2. 但你的数据说了更强的事

### 2.1 映射的上确界恰好是 1

一轮迭代对 `b` 做的是

```
b₁ = f(b₀) = k · b₀ · (2 − k · b₀)
```

`df/db = 2k − 2k²b = 0` → `b = 1/k`，`f(1/k) = 2 − 1 = 1`。

> **对任意 k、任意实数 b₀，`f(b₀) ≤ 1`。上确界是 1，和 k 无关。**

本轮 `k = 2/(1+0.046875) = 1.9104`，峰值在 `b₀ = 1/k = 0.5234`。

### 2.2 padding 槽正好坐在峰值上，所以 p50 就是峰值

`PADDING_FLOOR = 0.5`，而峰值在 0.5234。**padding 槽（占多数，决定 p50）映射到 ≈ 峰值。**

本机实测正是如此：

```
本机 iter01_b:  max 0.274   p50 0.2734    →  max / p50 = 1.00
```

**最大值就是 p50——所有槽都不超过 padding 的像，也就是峰值。这就是健康的样子。**

### 2.3 设备的 slot 0 是峰值的 5.64 倍

```
设备 iter01_b:  max 1.541   p50 0.2733    →  max / p50 = 5.64
```

**先确认标度一致**：设备 p50 = 0.2733，本机 p50 = 0.2734，**四位有效数字相同**。
（探针报的是密文，`_restore_magnitude` 把密文和 delta 同乘一个整数，
两边那个整数由 delta 决定、完全相同——p50 对上就证明了标度一致。）

于是：

> **设备的 slot 0 落在映射上确界的 5.64 倍处。**
> **没有任何实数 b₀ 能让 `k·b₀·(2−k·b₀)` 等于那个数。**

这比"slot 0 发散了"强得多。发散是"函数把它送去了很远的地方"；
**这是"结果不在函数的值域里"——那一步没有算出 `f`。**

（顺带排除一个读法：b₀ 为负会让 `f(b₀)` 为负、绝对值变大。但报的是 `max +1.541`，是正的。
正的 5.64 倍峰值，两头都够不着。）

### 2.4 而 `a` 走了完全相同的算子

同一个 `correction`（由 `b` 导出）、同一个 `_times`、同一个 `_restore_magnitude`。
`a` 在 slot 0 健康，`b` 不健康。

**所以不是某个算子整体坏了，是它对某个操作数在某个槽上算错了。**
`a` 起始是 `ones`（0/1），`b` 起始是自举后的分母——两者在 slot 0 的唯一区别是值本身。

---

## 3. 下一步：把 bracket 做成真的（已推）

`_trace_noise` 已经在 `_times` 之后、`_restore_magnitude` 之前打点，但它只报 `noise_level`，不报值。
现在那里加了两个**值**探针：

```
07.inv_iter01_a_times      ← _times 之后，_restore_magnitude 之前
07.inv_iter01_b_times      ← 同上
07.inv_iter01_a            ← _restore_magnitude 之后（原有）
07.inv_iter01_b            ← 同上
```

**一次跑就能把 `_times` 和 `_restore_magnitude` 用测量分开，而不是用假设。**

- `b_times` 的 slot 0 **已经**是 worst → 坏在 `_times`（multiply / relinearize / rescale）；
- `b_times` 正常、`b` 才坏 → 坏在 `_restore_magnitude`（conjugate-add / 整数乘），
  **这时你原来的结论才成立**，而且范围缩到两个算子。

请贴 `iter01` 和 `iter02` 的这四行。

---

## 4. 另一件事：`EvalScalarSub` 确实有调用方，我之前写错了

`SWEEP-changes-that-affect-other-workloads-20260914.md` §3.2 写着：
"bert-tiny 不调它（查过），**THOR 移植也不调**，所以目前没有已知调用方"。

**错了。** `he_inv` **每一轮迭代**都调：

```python
correction = self.subtract(2 / k * b.delta, b.ciphertext)   # 左操作数是标量
```

经 `pyfideslib/__init__.py:149` 派发到 `EvalScalarSub(double, CT)`——
**正是 `9bdd560` 里我改过符号的那个重载**，而且它就在现在出问题的这条路径上，每层几十次。

符号本身大概率是对的（设备上 `test_scalar_minus_ciphertext` 过，本机语义一致，
而且符号错会毁掉**每个**槽而不是一个）。**但"没有已知调用方"这条理由作废了**——
我在往 `dev` 推库代码时也是这么写的，那条 commit message 需要连同这里一起被读到。
上游之前必须按"有热路径调用方"来核实。已在 SWEEP 里原地更正。

---

## 5. 我这边改了什么

| 改动 | |
|---|---|
| `numeric.py` | `inv_iter{n}_{a,b}_times`：`_restore_magnitude` 之前的值探针，让 bracket 成立 |
| `SWEEP-...-20260914.md` | 更正"THOR 不调 `EvalScalarSub`"——`he_inv` 每轮都调 |
| 本报告 | 更正探针顺序的读法；给出"超出值域"的论证 |
