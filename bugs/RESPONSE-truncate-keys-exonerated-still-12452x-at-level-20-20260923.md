# truncate_keys 洗清了：关掉截断后 slot 0 仍然 12452x

日期：2026-09-23。针对 `37b3702`。

---

## 0. 结论

| 情形 | truncate_keys | level | slot 0 / p50 | 状态 |
|---|---|---|---|---|
| bootstrapped b | True | 36 | 1.1x | ✅ 干净 |
| bootstrapped b | True | 19 | 13072.7x | ❌ 坏 |
| bootstrapped b | **False** | 36 | 1.1x | ✅ 干净 |
| bootstrapped b | **False** | 19 | **12452.1x** | ❌ **仍然坏** |

**`truncate_keys=False` 没有修复 slot 0。** worst at 仍然是 0, 9891, 20170。截断洗清了。

---

## 1. 结果

### 1.1 truncate_keys=False, level 37 — 干净

```
truncate_keys=False, bootstrapped b at level 37  (product at level 36):
  slot 0      error 2.52191e-10   value 0.24073 against 0.24073
  slot 0 / p50 of the rest: 1.1x
  worst 19507(%16=3), 14851(%16=3), 15825(%16=1), 5236(%16=4), 26270(%16=14)
  -> TRUNCATION WAS THE CAUSE     ← 但这条是 level 37，本来就不会坏
```

### 1.2 truncate_keys=False, level 20 — 仍然坏

```
truncate_keys=False, bootstrapped b at level 20  (product at level 19):
  slot 0      error 1.37811   value 1.62225 against 0.244135
  slot 0 / p50 of the rest: 12452.1x
  worst 0(%16=0), 9891(%16=3), 1(%16=1), 20170(%16=10), 19782(%16=6)
  -> truncation exonerated; look at the bootstrap output
```

**worst at 0, 9891, 20170** — 和 truncate_keys=True 完全一样。误差 1.378 vs 1.439，同一个量级。

### 1.3 关键对比

| | truncate=True (level 19) | truncate=False (level 19) |
|---|---|---|
| slot 0 error | 1.439 | 1.378 |
| slot 0 / p50 | 13072.7x | 12452.1x |
| worst at | 0, 9891, 20170 | **0, 9891, 20170** |
| resident memory | 9356 MiB (11 truncated) | **9400 MiB** (0 truncated) |

**截断洗清。** 0 truncated keys, full memory，但 bug 完全不变。

---

## 2. 下一步

你说 §4：如果截断被洗清，下一个是"bootstrap 返回 scale degree 2，Engine.bootstrap 再 rescale 修回来"。

建议：在 test 里加一个 case，**bootstrap 后不做 rescale**（保持 scale degree 2），然后在 level 20 做 `_times`，看是否仍然坏。

如果 scale degree 2 是关键 → 问题在 bootstrap 产出密文的 scale degree
如果仍然坏 → 问题在 bootstrap 产出密文的其他结构（limb 布局、NoiseLevel）
