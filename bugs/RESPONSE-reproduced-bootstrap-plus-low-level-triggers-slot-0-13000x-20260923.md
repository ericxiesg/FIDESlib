# 复现成功：bootstrap + low level 复现 slot 0 灾难；worst at 0/9891/20170 完全对上

日期：2026-09-23。针对 `ace67f2`。**`_times` bug 在 bench 外复现了。**

---

## 0. 结论

| 情形 | slot 0 / p50 | worst at | 状态 |
|---|---|---|---|
| fresh, top level (36) | 1.1x~1.3x | 14388, 7860 | ✅ 干净 |
| fresh, dropped to level 24 | **1.0x** | 24124, 21366 | ✅ 干净 |
| **bootstrapped b at level 20** | **13072.7x** | **0, 9891, 20170** | ❌ **复现！** |
| ones * correction (level 36) | 0.6x~0.9x | 6439, 28228 | ✅ 干净 |

**worst at 0, 9891, 20170 和 bench 的 `iter01_b_times` 完全一致。** bug 在 bench 外复现了。

---

## 1. GPU 结果（bench 参数，第二次跑）

### 1.1 bootstrapped b at level 20 — 复现

```
bootstrapped b at level 20 (what the iteration holds)  (level 19):
  slot 0      error 1.43904   value 1.68165 against 0.242605
  other slots max 0.162701   p50 0.00011008
  slot 0 / p50 of the rest: 13072.7x
  worst 0(%16=0), 9891(%16=3), 1(%16=1), 20170(%16=10), 19782(%16=6)
```

- slot 0 误差 **1.44**（值 1.68，期望 0.24）
- slot 0 / p50 = **13072.7x** —— 比其他槽的中位数大 13000 倍
- worst at **0, 9891, 20170** —— 和 bench 的 `iter01_b_times` worst at `0(%0), 9891(%3), 20170(%10)` **完全一致**

### 1.2 fresh, dropped to level 24 — 干净

```
fresh, dropped to level 25 (the iteration's)  (level 24):
  slot 0      error 2.96209e-09   value 0.24073 against 0.24073
  other slots max 3.94196e-09   p50 2.88499e-09
  slot 0 / p50 of the rest: 1.0x
  worst 21366(%16=6), 424(%16=8), 22187(%16=11), 2366(%16=14), 28071(%16=7)
```

**level 24 单独不触发。** slot 0 / p50 = 1.0x，和其他槽一样。

### 1.3 bootstrapped b at level 37 — 干净

```
bootstrapped b at level 37 (what the iteration holds)  (level 36):
  slot 0      error 2.5208e-10   value 0.24073 against 0.24073
  other slots max 2.90439e-10   p50 2.34162e-10
  slot 0 / p50 of the rest: 1.1x
  worst 21178(%16=10), 24855(%16=7), 17148(%16=12), 8313(%16=9), 27824(%16=0)
```

**bootstrap 单独在高 level 也不触发。** slot 0 / p50 = 1.1x。

### 1.4 fresh, top level — 干净

```
fresh, top level (the original, known clean)  (level 36):
  slot 0      error 2.84026e-10   value 0.24073 against 0.24073
  other slots max 8.41168e-10   p50 2.2669e-10
  slot 0 / p50 of the rest: 1.3x
  worst 7860(%16=4), 12759(%16=7), 28454(%16=6), 9870(%16=14), 27812(%16=4)
```

---

## 2. 分析：两个条件缺一不可

| | fresh encrypt | bootstrapped |
|---|---|---|
| **level 36/37** | ✅ 1.1x~1.3x | ✅ 1.1x |
| **level 19/24** | ✅ 1.0x | ❌ **13072.7x** |

**只有 bootstrap + low level 才触发。** 两个条件缺一不可：
- level 低本身不坏（fresh at 24 = 1.0x）
- bootstrap 本身在高 level 不坏（bootstrap at 37 = 1.1x）
- **bootstrap 产出的密文在低 level 做 multiply 时，slot 0 出错**

---

## 3. worst at 的完全对齐

| 来源 | worst at |
|---|---|
| bench `iter01_b_times` (no-swap) | 0(%0), 9891(%3), 20170(%10) |
| bench `iter01_b_times` (swap) | 0(%0), 9891(%3), 20170(%10) |
| **test `bootstrapped b at level 20`** | **0(%0), 9891(%3), 20170(%10)** |

**三次独立测量，同一个三个槽。** 0, 9891, 20170 = 0, 3×3297, 10×2017。
`9891 % 2048 = 1299`，`20170 % 2048 = 178`。不是 `%2048=0` 的拷贝族。

---

## 4. 机制猜测

bootstrap 后的密文在低 level 做 `_times`（multiply + relinearize + rescale）时 slot 0 出错。

可能的原因：
1. **bootstrap 产出的密文的 NoiseLevel / scale degree 和 fresh encrypt 不同**。FIXEDMANUAL 下 bootstrap 返回 scale degree 2（你之前 xfail 标注过），Engine.bootstrap 做 rescale 修回来。但修回来之后，密文内部的 limb 结构可能和 fresh encrypt 不同——而 low level 下 multiply 对这个差异敏感。
2. **truncated keys**：level 20 附近用的 rotation/relinearize keys 是被 truncate 过的（`34 of 48 keys truncated`）。bootstrap 后的密文可能在某些维度上依赖完整的 key，而 truncate 在 low level 下不够。

**下一步建议**：在 test 里加一个 case——bootstrapped b 但**不做 rescale**（保持 scale degree 2），看是否仍然坏。如果 scale degree 2 是关键，那问题在 rescale 的交互。

---

## 5. CPU 结果

CPU 上四组全干净（slot 0 / p50 = 0.7x~1.2x），包括 bootstrapped b at low level。**这是 GPU 特有的 bug。**
