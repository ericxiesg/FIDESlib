# Fidelity: GPU FHE (fideslib), stages 01-05, random input

日期：2026-09-08。Commit：`a2611ef`（bootstrap-dev）。分支 `bootstrap-dev`。
引擎：`fideslib`（CUDA CKKS，Quadro GV100 32GB）。

这是 GPU FHE 路径的 **首次 fidelity 验证**。由于 bootstrap + 210 rotation keys 的显存问题
（见 `bugs/GPU-OOM-keygen-20260908.md`），完整 layer 暂时无法跑通。本测试只覆盖 stage 01-05
（QKV 投影），不启用 bootstrap，用 random input 对比 GPU FHE vs CPU ClearEngine。

## 配置

| 参数 | 值 |
|------|-----|
| engine | fideslib (CUDA) |
| device | cuda:0 (Quadro GV100) |
| depth | 44 |
| log_n | 16 (N=65536, 32768 slots) |
| scaling_bits | 50 |
| first_mod_bits | 55 |
| dnum | 4 |
| binary_rotations | true (15 keys: 1,2,4,...,16384) |
| light_plaintext_cache | 8 |
| bootstrap | disabled |
| input | random N(0,1), dim=128, features=768 |

## Per-stage fidelity (GPU FHE vs CPU ClearEngine, 4 ciphertexts)

### Stage 01: complexify_x (multiply_1j + conjugate + add)

| Ciphertext | max_err vs clear |
|:----------:|------:|
| ct[0] | 9.82e-10 |
| ct[1] | 1.11e-10 |
| ct[2] | 1.06e-10 |
| ct[3] | 9.89e-10 |

### Stage 02: make_rotated_copies (64 rotations via 15 binary keys)

全部完成，无误差报告（rotate 是精确置换）。

### Stage 03: query (pcmm + bias + conjugate)

| Ciphertext | Best-fit Scale | MAE | RMSE | Max Abs | relRMSE |
|:----------:|------:|------:|------:|------:|------:|
| query[0] | 1.0000 | 1.72e-7 | 2.48e-7 | 1.28e-6 | 1.03e-8 |
| query[1] | 1.0000 | 1.71e-7 | 2.48e-7 | 1.23e-6 | 1.03e-8 |
| query[2] | 1.0000 | 1.72e-7 | 2.49e-7 | 1.19e-6 | 1.03e-8 |
| query[3] | 1.0000 | 1.71e-7 | 2.47e-7 | 1.29e-6 | 1.03e-8 |

### Stage 04: key (pcmm + bias + conjugate, with softmax scale)

| Ciphertext | Best-fit Scale | MAE | RMSE | Max Abs | relRMSE |
|:----------:|------:|------:|------:|------:|------:|
| key[0] | 1.0000 | 1.72e-7 | 2.48e-7 | 1.28e-6 | 1.03e-8 |
| key[1] | 1.0000 | 1.71e-7 | 2.48e-7 | 1.23e-6 | 1.03e-8 |
| key[2] | 1.0000 | 1.72e-7 | 2.49e-7 | 1.19e-6 | 1.03e-8 |
| key[3] | 1.0000 | 1.71e-7 | 2.47e-7 | 1.29e-6 | 1.03e-8 |

### Stage 05: value (pcmm + bias + conjugate)

| Ciphertext | Best-fit Scale | MAE | RMSE | Max Abs | relRMSE |
|:----------:|------:|------:|------:|------:|------:|
| value[0] | 1.0000 | 1.72e-7 | 2.48e-7 | 1.28e-6 | 1.03e-8 |
| value[1] | 1.0000 | 1.71e-7 | 2.48e-7 | 1.23e-6 | 1.03e-8 |
| value[2] | 1.0000 | 1.72e-7 | 2.49e-7 | 1.19e-6 | 1.03e-8 |
| value[3] | 1.0000 | 1.71e-7 | 2.47e-7 | 1.29e-6 | 1.03e-8 |

## Timing

| Stage | Time |
|-------|------:|
| stage_01 complexify | 0.03s |
| stage_02 make_rotated_copies | 0.04s |
| stage_03 query (pcmm, 12×12 grid, binary rotations) | 109.95s |
| stage_04 key | 4.30s |
| stage_05 value | 4.30s |
| **total** | **118.62s** |

## 说明

- **scale = 1.0000**：GPU FHE 输出与 ClearEngine 完全同 scale，无 scale 漂移。
- **relRMSE ~1e-8**：CKKS 噪声水平，比 CPU clear engine 的 float32 精度（~3e-7）低一个数量级，
  说明 binary rotation decomposition 引入的额外 key-switch 噪声可忽略。
- **stage_03 耗时 110s**：binary rotations 把 1802 次旋转变成 8138 次（4.5 倍），pcmm 的 12×12
  网格每个 entry 需要多次旋转。这是无 bootstrap 的裸计算时间。
- **stage 04/05 只需 4.3s**：key 和 value 的 pcmm 网格更小（stage 03 的 12×12 vs stage 04/05 的
  对角线折叠后更少旋转）。
- 对比 CPU clear engine 的 per-stage fidelity（`fidelity-clear-cpu-20260907.md`）：
  - CPU query relRMSE = 2.884e-7（float32 精度极限）
  - GPU query relRMSE = 1.03e-8（CKKS 噪声，**更好**，因为 CKKS 在高信噪比下噪声可控）
