# 真实 checkpoint 下 stage 15 送进 bootstrap 的是 103.9，是界的 325%

日期：2026-09-16。**这条独立于精度那条线，而且我先前报的数低估了 6 倍。**

---

## 1. 先更正我自己

我在 `RESPONSE-bootstrap-precision-20260915.md` §4.1 报过"stage 15 的 bootstrap
输入是 17.57"。**那是用随机权重（std 0.04）量的，不是真实模型。**

用真实 checkpoint（`textattack/bert-base-uncased-MRPC`，417 MB，本地已缓存）
和真实 embedding 重跑，整层在 `ClearEngine` 的**精确算术**下：

```
tokens 23/128   embedding |x| max 11.14  p50 0.2755

thorfhe.clear.ScaleMismatch: bootstrap: the message reaches 103.9, which is 325%
of q0/Delta = 32. ...
  layer.py:280 -> layernorm.py:172 -> stages.py:203
```

**103.9，不是 17.57。低估了 6 倍。**

精确算术意味着这不是噪声——**103.9 就是那个残差的真实值**。

---

## 2. 为什么这是"替换"不是"精度差"

CKKS bootstrap 的 ModRaise 之后，密文携带 `m + q0*I`，靠一个只在零附近近似模约简的
正弦把 `m` 取回来。约束是 `|m| < q0/2`，即 `|v| < q0/(2*Delta) = 16`。

`|v| = 103.9` 是这个约束的 **6.5 倍**。`m/q0 = 3.25`，正弦绕了三圈多——
**恢复出来的值和原值无关，不是"误差大"，是另一个数。**

（我的守卫用的是更宽松的 `q0/Delta = 32`，已经放宽了 2 倍，仍然超 3.25 倍。
所以这不是阈值松紧的问题。）

---

## 3. 而 THOR 的代码和我们**逐行相同**

`THOR/src/thor/he.py:1448-1461`：

```python
def stage_15_prepare_layernorm(self, x, y):
    output = np.full((8,), None, dtype=object)
    for index in range(8):
        output[index] = self.add(x[index], y[index])
    for index in range(4):
        temp = self.add(output[index], self.multiply_1j(output[index + 4]))
        temp = self.bootstrap(temp)          # <- 没有减半，和我们一样
        temp = self.level_down(temp, 3)
        ...
```

**没有减半，直接 bootstrap。** 我们是忠实移植。

而且 `THOR/src/thor/data_encoder.py:41` 的 `encrypt_embedding` 对 embedding
**没有任何缩放**——原样打包加密。所以 THOR 送进 stage 15 的幅度和我们一样是 ~100。

---

## 4. 于是有一个矛盾，必须解开

EasyFHE THOR 的参数是 `RESCALE_PRIME_BITS=59, FIRST_PRIME_BITS=60`，
即 `q0/Delta = 2`，而且 `SECRET_KEY_DIST = "SPARSE_TERNARY"`
（`message_scaling_factor = 1.0`，不像 UNIFORM 那样有 512 倍的内部缩放）。

**在界为 2 的参数下 bootstrap 一个 ~100 的值**——按上面的教科书约束这不可能。

所以只有两种可能：

* **(a) 教科书那个 `|v| < q0/(2*Delta)` 不描述这些实现**，desilofhe（以及可能 FIDESlib）
  在内部另有缩放，实际可用范围远大于 `q0/Delta`；
* **(b) 我们的 stage 15 在真实模型上本来就是坏的**，
  而 softmax 那条线一直挡在前面，所以没人走到这里。

**两种都必须知道，而且一个实验就能分开。**

---

## 5. 三行的判据

在设备上 bootstrap 一个已知常数，幅度取 103.9：

```python
import numpy as np
v  = 103.9
ct = engine.encrypt(np.full(engine.slots, v, dtype=complex))
out = np.real(np.asarray(engine.decrypt(engine.bootstrap(ct))))
print("in", v, "out", out[:4], "abs err", np.max(np.abs(out - v)))
```

再用 `v = 1.0` 和 `v = 10.0` 各跑一次做对照。

* **103.9 出来还是 ~103.9（误差和小值同量级）** → (a)：实现的可用范围远大于 `q0/Delta`，
  那么我加的那个幅度守卫**阈值定错了**，我来改，并且 stage 15 没问题；
* **103.9 出来是个无关的数（比如落在 [-16, 16] 里）** → (b)：**stage 15 在真实模型上是坏的**，
  这是一个独立于精度的第二个 bug，而且在 `q0/Delta = 2` 下会更糟（超界 52 倍）。

如果是 (b)，那么"迁到 EasyFHE 参数"这件事在 stage 15 的幅度处理解决之前**不能做**——
不是因为它不好，而是因为它会把一个已经坏的地方变得更坏。

---

## 6. 这条和精度那条的关系

**它们是两个独立的问题**，而且这一条在流水线上更靠后：

```
stage 07 softmax   <- bootstrap 精度不够（1.04e-2 vs 待求逆量 4.88e-4），已定位到 stage 4
stage 15 layernorm <- bootstrap 输入超界 3.25 倍（本文）
```

设备上的现象从 softmax 开始坏，所以 stage 15 这条**还没有被观察到**——
它被前面那条挡住了。修好 softmax 之后它会立刻浮上来。

**所以值得现在就测第 5 节那三行**，免得修完 softmax 再撞一次。

---

## 7. 我这边接下来

如果是 (b)，我会去查 `norm_1` 和 `output_dense` 各自的幅度，
看 ~100 是 BERT 残差本来的量级（那就要在 THOR 的调度里找它本来该怎么被压住），
还是我们某一处的缩放和 THOR 不同。

`ClearEngine` 上用真实 checkpoint 就能量，不需要 GPU。
