# 明文缓存的 key 是整份数组，12 层要 112 GiB 宿主内存

日期：2026-09-18。承接 `7a03a0d`。**这条是 `--layers 12` 的拦路石，不影响 `--layers 1`。**

---

## 0. 结论

`Stages.plaintext`（改之前）：

```python
key = (value.shape, value.dtype.str, value.tobytes())
cached = self._plaintexts.get(key)
```

`tobytes()` 的结果**被留在 key 里**。于是每条缓存项存两份 512 KiB：
一份 light plaintext（缓存的值），一份 key。

一层 19,137 个不同数组（就是你量到的 encode 次数），所以：

| | 每层 | 12 层 |
|---|---|---|
| key 占的宿主内存 | **9.34 GiB** | **112 GiB** |
| light plaintext 本身 | 9.34 GiB | 112 GiB |

而且 `self._plaintexts` **从建到死没有任何清理**——没有容量上限，没有 evict，
`ClearLightPlaintextCache` 绑定了但 Python 侧一次都没调过。

`--layers 1` 撑得住（你那台机器内存够），**`--layers 12` 会在宿主端 OOM**。

---

## 1. 改法：key 存摘要，不存内容

```python
key = (value.shape, value.dtype.str, hash(value.tobytes()))
```

同样是 `tobytes()` 再哈希——**和原来一样的计算**，只是不把那 512 KiB 留下。

本机实测（512 KiB complex128 数组，400 次）：

| | 每次 | 每层 19,137 次 | key 保留 |
|---|---|---|---|
| 原来 | 183 µs | 3.51 s | 512 KiB |
| 现在 | 221 µs | 4.23 s | **8 字节** |

**9.34 GiB → 0.00014 GiB**，代价 +0.7 s/层。

试过 `blake2b`（128 位，更安全），969 µs/次 = **18.5 s/层**，比 Python 自带的 siphash 慢 4 倍，
不值，放弃了。

### 代价说清楚

64 位摘要意味着**精确比较换成了概率比较**。一层 19,137 条，两个不同数组撞同一个摘要的概率
约 **1e-11**。比这条流水线其他出错方式低几个数量级，但不是零。

**真正的解法是根本不要这张缓存**：让 `encode_layer` 直接往下传 LightPlaintext
（`he.LightWeights` 已经是这个形状），热路径就只递句柄。那之后 key 的问题不存在。
现在这个改动是在那之前不让 12 层跑不起来。

---

## 2. 和 `7a03a0d` 的关系

两条是独立的，都要：

| | 治什么 | 量级 |
|---|---|---|
| `7a03a0d` 两 tower 编码 | 编码**时间** | 1109 s → ~58 s / 层 |
| 本条 摘要 key | 缓存**内存** | 9.34 GiB → 0 / 层 |

`7a03a0d` 不减少编码次数，所以缓存条目数不变，内存问题照旧。
本条不加快编码，所以时间问题照旧。

---

## 3. 请在设备上确认

1. `7a03a0d` 能编译，`relRMSE` 逐位不变，`--time-ops` 里 `encode_to_light_plaintext` 掉到多少。
2. **`--layers 2` 或 3**，看宿主内存（不是显存）的峰值。改之前每层涨约 18.7 GiB
   （key + light plaintext），改之后应该只涨约 9.34 GiB。
3. 如果宿主内存仍然是问题，下一步就是 §1 末尾说的那条：`encode_layer` 直接产 LightPlaintext，
   并给 `_plaintexts` 加一个容量上限。

200 passed。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
