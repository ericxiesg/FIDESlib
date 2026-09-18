# 编码结果存盘：`--plaintext-cache DIR`

日期：2026-09-18。承接 `a56aad6`（encode 占 96.2%）。

---

## 0. 结论

权重编码**不随 sample 变、也不随跑次变**，但现在每跑一次都重编一遍。存盘复用：

```
python -m thorfhe.bench fhe --engine fideslib ... --plaintext-cache /path/to/cache
```

第一次跑照常编码并落盘，之后每次从盘上读。

| | 每层 | 12 层 |
|---|---|---|
| 磁盘占用 | **约 9.3 GiB** | **约 112 GiB** |
| 冷跑（编码 + 写盘） | 1109 s + 约 19 s | |
| **热跑（只读盘）** | **约 19 s** | |

按 500 MB/s 估的 I/O。你有 300 GB，12 层放得下还有余量。

---

## 1. 为什么按"出处"做键，不按内容

内容寻址（哈希数组当文件名）是直觉做法，但在这里是错的：**稳定的哈希必须读完每个字节**。
本机实测 512 KiB 的数组：

| | 每次 | 每层 19,137 次 |
|---|---|---|
| sha1 | 648 µs | **12.4 s** |
| blake2b-128 | 1018 µs | 19.5 s |
| crc32 | 73 µs | 1.4 s，但 32 位——19,137 条目下碰撞概率约 **4%**，不能用 |

而一个权重的身份本来就是已知的：**checkpoint、第几层、哪个 field、在 field 里的第几个**，
四样都是免费的。所以键是出处，不是内容。

`Stages.plaintext` 的内容缓存**保留**，它服务的是 mask——mask 在调用点被重建，
确实需要按内容认。实测一层 21,195 次带数组的乘法里，mask 命中约 2,058 次。

---

## 2. 关键设计：传的是 thunk，不是数组

`store(layer_index, field_name, build)` 收的是**构造函数**，不是构造好的数组。
所以命中时**两样都不做**：不编码，也不构造那 9.7 GB 的 numpy。

这也是为什么挂在 `encode_layer` 的 field 上，而不是挂在 `Stages.plaintext` 上——
后者只能看到**已经构造好**的数组，省不掉构造。

### 附带解决了宿主内存

field 现在直接是 light plaintext，不是 ndarray，所以 `Stages.plaintext` **原样透传、不入缓存**。
我在 `e605fac` 里提的那 9.3 GiB/层的 `_plaintexts`（条目半 MiB、永不清理、12 层 112 GiB）
在热路径上就不再产生了。

---

## 3. 目录布局与安全性

```
DIR/v1/<tag>/layer00/query/000000.flpt
                          /000001.flpt
                          /tree.json      <- 最后写，原子替换
```

`tag` 由**所有会改变编码结果的东西**拼成：

```
<model>-n16-d37-sb59-fmb60-rs256-srs16
```

`residual_scale` / `score_refresh_scale` 是被 `encode_layer` **烤进明文**的，
ring/scaling 参数决定系数——所以换了参数就是另一个目录，**不可能读回不匹配的缓存**。

`tree.json` 记录 field 的结构（嵌套的 object array 形状、以及折进去的标量），
**最后写且原子替换**：它存在就意味着旁边每个 leaf 都是完整的。跑到一半被 kill 会留下
没有 manifest 的目录，下一次直接重建。manifest 损坏或 leaf 缺失会**打印一行然后重建**，
不会让缓存问题看起来像模型坏了。

---

## 4. 验证

本机没有 CUDA，所以用一个假的 light-plaintext 引擎验的：

- **真实 THOR geometry 的 `encode_layer` 往返**：`attention_norm` + `output_norm` 两个 field，
  32 个 leaf **逐位相同**，第二遍 **0 次编码**。
- `tests/test_plaintext_store.py` 9 个用例：往返一致、命中不跑 thunk、layer/field 不串、
  无 manifest 重建、manifest 损坏重建、leaf 缺失重建、tag 隔离、
  ClearEngine 不启用、**八个 field 全部走 store**（漏一个就是每跑一次重编一次）。
- `bench fhe --engine clear ... --plaintext-cache DIR`：ClearEngine 正确拒绝（不建目录），
  `relRMSE 2.776e-03` 不变。

209 passed, 67 skipped。

---

## 5. 请在设备上

```
# 第一次：编码 + 落盘
python3 -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 --time-ops \
  --refresh-after-dense --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16 \
  --compact --extra-rotation-keys 6 --plaintext-cache /data/ptcache

# 第二次：同一条命令，应该看到
#   plaintext cache ...: 8 fields loaded, 0 encoded
#   --time-ops 里 encode_to_light_plaintext 降到 0 次
```

1. 两次的 `relRMSE` 必须**完全一致**——不一致就是落盘/读回丢了东西，请立刻报。
2. 第二次的 `--time-ops` 表，特别是 `encode_to_light_plaintext` 那一行。
3. `du -sh /data/ptcache`，对一下 9.3 GiB/层。

和 `7a03a0d`（两 tower 编码）是**叠加**的：那个让冷跑的编码从 1109 s 降到约 58 s，
这个让热跑根本不编码。两个都上之后，一层的编码成本第一次约 58 s，之后约 19 s 的读盘。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
