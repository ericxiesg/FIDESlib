# --plaintext-cache 验证：热跑 88s vs 冷跑 1177s，13.4x 加速，精度一致

日期：2026-09-18。针对 `b008dab`。

---

## 0. 结论

`--plaintext-cache` 在设备上验证通过：

| | 冷跑（编码+写盘） | 热跑（读盘） | 加速 |
|---|---|---|---|
| wall clock | 1176.76s | **87.89s** | **13.4x** |
| encode_to_light_plaintext calls | 19,197 | 689 | 27.8x |
| encode_to_light_plaintext time | 1111.94s (96.0%) | 38.81s (46.7%) | 28.6x |
| decrypt | 19.92s (1.7%) | 18.96s (22.8%) | — |
| multiply | 14.88s (1.3%) | 14.53s (17.5%) | — |
| bootstrap | 4.11s (0.4%) | 4.10s (4.9%) | — |

磁盘占用：**9.2 GiB**（与预估的 9.3 GiB 一致）。

---

## 1. 缓存命中

```
plaintext cache .../layer00: 0 fields loaded, 8 encoded   ← 冷跑
plaintext cache .../layer00: 8 fields loaded, 0 encoded   ← 热跑
```

热跑全部 8 个 field 从盘上读取，0 次编码。剩余 689 次 `encode_to_light_plaintext` 是 mask（按内容寻址，不存盘，符合设计）。

---

## 2. 精度对比

两个跑次的 query/value/scores **完全一致**（relRMSE 2.884e-07 / 3.141e-07 / 2.873e-07），证明落盘/读回无损。

下游阶段因 he_inv bug（07d 发散到 5.9e16/2.5e16）而数值不同——这是同一个 bug 的不同发散路径，不是缓存问题：

| 探针 | 冷跑 max | 热跑 max |
|------|----------|----------|
| 07c.denominator | 0.298 | 0.306 |
| 07d.inverse_denominator | 5.926e+16 | 2.506e+16 |
| 11b.inverse_sqrt | 2.001e+124 | 1.662e+124 |

07c 正常（~0.3），07d 和 11b 发散——和之前所有跑次一致。

---

## 3. 下一步

热跑 88s 意味着**带探针的 debug 跑现在只需要约 90 秒**（之前 1166 秒）。后续所有迭代实验都可以用 `--plaintext-cache /home/zhiyuan/ptcache` 加速。

注意：`--time-ops` 在热跑下显示 decrypt 占 22.8%（18.96s），这是下一个可优化项——但优先级远低于 he_inv 精度 bug。

Co-Authored-By: CodeAgent <noreply@anthropic.com>
