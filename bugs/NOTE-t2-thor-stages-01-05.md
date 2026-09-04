# T2：THOR stage_01–05 移植完成，需要在真机上跑 FHE 那一层

设计与取舍写在 `docs/thor_port.md`，这里只讲**怎么验**和**哪些地方我没底**。

新增 `python/thorfhe/` 包和 `python/tests/test_stage6_thor_qkv.py`。numpy 那部分我在本地跑通了
（下面有结果）；FHE 那部分本机没有引擎，**没跑过**。

## 先说两个查出来的真问题

### 1. 旋转方向和 desilofhe 是反的

desilofhe 的 `rotate(ct, delta)` 是把 slot `i` 搬到 `i + delta`；OpenFHE / FIDESlib 的
`EvalRotate(ct, delta)` 是搬到 `i - delta`。**符号相反。**

这不是猜的：把整条 QKV 流水线用 numpy 在 slot 空间实现之后，取 `W = I`、`X` 只有一个非零元做探针，
用错方向时不同的 `X[t,f]` 会落到同一个输出 slot（信息丢失，映射不是单射）；用对方向时 42 个探针
落在 42 个不同 slot 上。改对之后整条流水线和 `X @ W.T + 2b` 精确相等（1e-16）。

`thorfhe/stages.py` 里统一在 `Stages.rotate` 取一次负号，所以 stage 代码里的每个 delta 都和
`he.py` 逐字一致。**后面移植 stage_06 以后的部分时，delta 照抄 he.py 就行，不要自己再取负。**

### 2. bias 是 2 倍

`encode_weight` 里有个 `/2`，用来抵消结尾 `y + conj(y)` 的加倍；但 THOR 的 `encode_b` 没有除 2。
所以 QKV stage 的输出是 `X @ W.T + 2*b`，不是 `+ b`。这是 THOR 自己的口径（或者是 THOR 的一个
潜在 bug），我**照搬没改**，因为后面的 stage 是按 THOR 的数值标定的。测试里写的就是 `+ 2*b`。
如果后面对拍真实 MRPC 精度时发现不对，回来看这一条。

## 一个需要你判断的分歧：FIXEDMANUAL vs FIXEDAUTO

`he.py` 里有两处在 FIXEDMANUAL 下是**非法**的：

- `prepare_for_multiply(x) = ntt(rescale(x))`：对一个已经是 canonical（scale Δ）的密文再 rescale，
  scale 会掉到 Δ/q。
- `rotate_internal` 里 `masked = multiply(mask, x)`（scale Δ²）然后 `subtract(x, masked)`（x 是 Δ）——
  两个 scale 不同的密文相减。

这两处合起来说明 **desilofhe 是自动管理 scale 的**（类似 OpenFHE 的 FLEXIBLEAUTO/FIXEDAUTO），
`he.py` 里的 `rescale` / `level_down` 更像是 level 调度提示而不是严格的 FIXEDMANUAL 记账。

我的处理：在 FIXEDMANUAL 下改写这两处（`prepare_for_multiply` 变成恒等；`rotate_internal` 改成
乘 mask 和乘 (1-mask) 两次再一次 rescale），数值完全等价，每次多一次明文乘法。
`ClearEngine` 会严格检查 level 和 scale，不匹配就抛 `ScaleMismatch`，所以这类错误在毫秒级测试里
就会暴露，不用等 GPU 上的噪声。

**已定：继续用 FIXEDMANUAL**，不切 FIXEDAUTO。所以上面那两处改写是长期方案，stage_06 及以后
也按同样的规矩写：

- 明文乘法之后必须 rescale 才能和别的 canonical 密文相加/相减；
- 需要"掩掉一部分"时用 `mask` 和 `1-mask` 各乘一次，不要 `x - mask*x`；
- `he.py` 里对 canonical 密文的 `rescale` 是 desilofhe 的延迟结算，移植时删掉；
- `level_down` 照抄（那是 THOR 主动的 level 调度，不是 scale 记账）。

`ClearEngine` 会把违反上面规矩的地方抛 `ScaleMismatch`，所以先在 numpy 层跑通再上 FHE。

## 本地已经跑通的部分（numpy，无引擎）

```
qkv_computes_xw_plus_bias[bert]  : 32768 slots, max err 7.8e-16   (~3 s)
qkv_computes_xw_plus_bias[small] :  4096 slots, max err 1.1e-16
unused_slots_stay_empty          : padding slot 全 0
level_schedule                   : stage 01–05 共耗 8 level（THOR 主动 level_down 6 + 2 次 rescale）
scale_discipline_is_enforced     : ClearEngine 能抓住 he.py 那种 mask-and-subtract
rotation_plan_covers_every_rotation
```

**两个几何都过**是关键：`SMALL` 故意设 `pack=4, n_slot=8`，而 THOR 里这两个数都是 16。
如果我把这两个量混为一谈，生产尺寸照样过、小尺寸会挂。

## 需要你在真机上跑的

```bash
export PYTHONPATH=$PWD/python
PYFIDESLIB_DEVICES=cpu    pytest python/tests/test_stage6_thor_qkv.py -x -v
PYFIDESLIB_DEVICES=cuda:0 pytest python/tests/test_stage6_thor_qkv.py -x -v
# 然后全量回归
PYFIDESLIB_DEVICES=cpu    pytest python/tests -x -v
PYFIDESLIB_DEVICES=cuda:0 pytest python/tests -x -v
```

FHE 那三个用例（`test_fhe_stages_match_clear` / `test_weight_files_round_trip` /
`test_fhe_key_and_value_reuse_the_same_input`）用 4096 槽（log_n=13、depth=12），只有 48 次明文乘法、
14 次旋转，CPU 上应该也很快。

判据：
- `test_fhe_stages_match_clear` 把 FHE 的解密结果和 numpy 镜像逐 slot 比，阈值 1e-5；再把解码结果和
  `X @ W.T + 2b` 比。**这是主判据。**
- 三个用例都会先断言 level 和 numpy 镜像一致。level 对不上说明 `EvalLevelReduce` / `Rescale` 的记账
  和我假设的不一样，请把实际 level 贴进 report。

## 我没底的地方（按可能性排序）

1. **`Engine.multiply(ct, light_plaintext)` 在 ct 处于非顶 level 时**。stage 里明文乘法发生在
   level = depth-6 和 depth-7，light plaintext 要按密文 level 展开。T1 的 `test_expands_at_the_ciphertext_level`
   覆盖了这个，但没覆盖"同一个 light plaintext 在一次运行里被多个不同 level 的密文使用"——
   mask 就是这种用法（`rotate_internal` 里所有 delta 的 mask 都在同一 level，应该没事）。
2. **`Engine.add_inplace`**：`parallel_diagonal_pc_mult` 里累加 48 次，用的是 `EvalAddInPlace`。
   如果它对 GPU 句柄的原地语义和我想的不一样（比如返回新对象而不改原对象），累加结果会错，
   表现为输出数值偏小很多。
3. **精度**。48 项累加、scaling 50 bit、值 ~0.1，1e-5 应该很宽松。如果只是差一点点（比如 3e-5），
   把阈值放到 1e-4 并在 report 里记下实测值，不要动算法。
4. **`log_n=13` + `depth=12` 的 4096 槽几何**：`Engine.slots == 4096`，测试开头有断言。

## 下一步（T3）

stage_06（转置、make_copies、Score CC-MM）、stage_07 softmax、stage_08 Context CC-MM。
这些要用到 `masks["transpose"]` / `masks["ccmm"] / `masks["make_copies"]`，`encode_weights.py` 里
那几段 mask 生成我还没搬（`pre_encode_masks` 的 transpose/ccmm 部分）。
