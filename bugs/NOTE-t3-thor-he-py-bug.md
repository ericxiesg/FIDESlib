# T3：stage_06/08 移植完成，并在 THOR 的 he.py 里发现一个真 bug

代码在 `python/thorfhe/attention.py`，测试 `python/tests/test_stage7_thor_attention.py`（10 个用例，
本机全过）。设计写在 `docs/thor_port.md`。这份 note 只讲那个 bug 和怎么复核。

## 结论

**`THOR/src/thor/he.py` 的 `stage_06_attention_score` 算错了 attention score。**

误差约为 score 自身量级的 **32%**，集中在 8 个输出密文里的第 1、2、5、6 个。不是数值精度问题，
是路由表写错。

## 为什么这么说

CC-MM 把 64 项内积累加进「每个输出密文 4 个累加器列」。一个贡献落在第 2-3 列还是第 0-1 列，
取决于源密文下标绕回了几次。规则是

```
column = 2 * ((offset // out_dim) % 2)        offset = i - (in_index // pack)  等
source = offset % out_dim
```

三条独立证据说明这就是正确规则：

1. **`he.py` 自己的 `in_index % pack != 0` 分支就是这个规则**（那里 `out_dim=4`、offset ∈ [-4,3]，
   等价于 `offset < 0`）。
2. **`he.py` 的 `stage_08_attention_context` 两个分支都是这个规则**，而且那里 `out_dim=2`、
   offset 能到 -4，会出现「绕两次又回到第 0-1 列」的情况 —— 它的表里确实是这样。
   `test_accumulator_routing_reproduces_he_py_stage_08` 逐条比对了 stage_08 的 8 张表，全中。
3. **只有 `stage_06` 的 `in_index % pack == 0` 分支不是**：它按 `i == 0` 分流。这与规则在
   `in_index // pack == 1` 时一致，在 2 和 3 时不一致 —— 也就是 `in_index = 32` 和 `48`。

数值上：把每个 `in_index` 单独跑出来和它的闭式对比（`make_copies[n]` 恰好提供
`d = (n + tau) % n_out` 那一项，所以单项贡献可以写下来），**64 项里 62 项精确相等，
只有 32 和 48 不对**，而这两个正是 `j == 0 且 block ∈ {2,3}`。换成正确规则后，
整个 stage_06 与 `Q K^T` 精确相等（3e-16）。

## 需要你复核的

我是拿**线性代数本身**当基准的（`decode(stage_06(Q,K)) == Q K^T` 逐槽相等），不是拿 THOR 自己的
输出。草稿 OpenFHE 仓库不在这个 checkout 里，所以我没法和 THOR 的实际数值对拍。

如果 THOR 论文报的 MRPC 精度是真的，32% 的 attention score 误差不该活得下来。所以两种可能：

- 论文/artifact 在这一处确实有 bug（那我们的修正是对的，而且值得反馈给作者）；
- 或者下游有什么补偿，我没看出来。

**如果服务器上能拿到 THOR 原始仓库或那份草稿 C++ 输出，麻烦跑一次对拍**：同样的 Q、K，
比较 `he.py` 的 stage_06 输出和 `Q K^T`。这能一锤定音。

代码里 `AttentionScore.accumulator_column` 是可覆盖的，`test_he_py_accumulator_table_is_wrong`
用 he.py 的路由跑一遍并断言它明显错、且只错在那 4 个密文上 —— 所以将来有人「修回」原表会立刻失败。

## 这一批还做了什么

| 操作 | 契约（逐槽精确，零容差） |
|---|---|
| `transpose_upper_to_lower` | `out[ct][g,tau,b] = k[t,f]`，`ct*pack+g = (t-d) mod n_out`，`tau = d + n_out*((d > t mod n_out) XOR (t // n_out))` |
| `make_copies` | `copies[l][g,t,b] = q[t, n_out*b + (l+t) mod n_out] / 2`，对所有 g |
| `stage_06_attention_score` | `out[ct][g,tau,b] = (Q_b K_b^T)[tau, (ct*pack+g+tau) mod dim]` |
| `stage_08_attention_context` | `out[ct][g,tau,b] = ctx[tau, n_out*b + (ct*pack+g+tau)%n_out] + 1j*ctx[tau, n_out*b + ((ct+2)*pack+g+tau)%n_out]`，`ctx = A_b V_b` |

这些契约都是拿 one-hot 探针从 numpy 模型里反推出来的，然后用随机数据逐槽验证。

FIXEDMANUAL 的规矩照 T2 那三条办；`ClearEngine` 严格校验 level 和 scale，所以这些用例同时也在
验证 level 调度。stage_06 耗 4 level，stage_08 耗 2 level（不含它后面的 bootstrap）。

## 跑法

```bash
export PYTHONPATH=$PWD/python
pytest python/tests/test_stage7_thor_attention.py -q      # 不需要编译扩展，纯 numpy
```

这批用例**不依赖 pyfideslib 扩展**（conftest 现在允许扩展缺失时跳过），所以在任何机器上都能跑。
FHE 那一层（log_n=16）没有做，CPU 上太慢；等 stage_07 补齐、能端到端跑一层时再一起上 GPU。

## 还没做

stage_07 softmax（Stockmeyer 多项式、he_exp1/2、he_inv、update_inv_D、bootstrap）——
是剩下最大的一块，而且是近似运算，验证方式要换成误差阈值而不是逐槽精确。

---

## 补充（stage_07/08 完成后）

T3 全部完成。`python/thorfhe/` 现在有 stage_01–08。numpy 引擎（严格 FIXEDMANUAL）上的实测：

| 检查 | 结果 |
|---|---|
| stage_06 = `Q K^T` | 3.3e-16（逐槽） |
| stage_08 = `A V` | 3.1e-16（逐槽） |
| stage_07 softmax 行和 | 1.0023（`output_alpha = 0.01` 之内） |
| stage_07 权重 vs 真 softmax | max 2.6e-3，mean 1.9e-5 |
| stage_06→07→08 vs `softmax(QK^T)V` | 相对误差 < 2% |

`pytest python/tests -q` 在没有编译扩展的机器上是 **33 passed, 40 skipped**。

### 两个坑，后面接手的人务必先看 `docs/thor_port.md`

1. **指数记账**：`he_exp1` 给的是 `exp(u/2)`（u 是喂给 `he_softmax` 的值），`he_exp2` 是 `exp(u/4)`
   （量程宽一倍所以斜率减半），每次 `update_inv_D` 把指数翻倍，而 `stage_07_softmax` 的
   bootstrap 折叠会把 score **翻倍**。走窄多项式还是宽多项式由 `max_x >= 30` 决定，
   所以重新标定时如果 `l` 不跟着动，softmax 的温度会悄悄变。
2. **softmax 输入窗口两边都很窄**：超过 `max_x` 一成，15 次多项式就发散、分母溢出；
   低于 `inv_epsilon`，Goldschmidt 没有可收敛的东西。可用的分母范围只有约三个数量级 ——
   key 投影里的 `softmax_scale = 1/512`（第 2 层 1/1024）就是为了把 score 打进这个窗口。
   `thorfhe.softmax.calibrate(scores)` 可以按实际分布重算这三个参数，T5 上真实 MRPC 数据时要用。

### 还是那个请求

如果服务器上能拿到 THOR 原始仓库或草稿 C++ 输出，麻烦对拍一次 stage_06：同样的 Q、K，
比较 `he.py` 的输出和 `Q K^T`。这是唯一能区分「THOR 有 bug」和「THOR 下游有补偿」的办法。
