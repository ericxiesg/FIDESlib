# Fidelity: CPU Clear Engine, 1 layer, 4 MRPC samples

日期：2026-09-07。Commit：`4e34444`（bootstrap-dev）。分支 `bootstrap-dev`。
引擎：`clear`（`ClearEngine`，精确算术 + FIXEDMANUAL level/scale 契约，无 CKKS 噪声）。
硬件：CPU（远程服务器，非 GPU）。

这是 THOR-v2 移植的 **代数正确性基线**：`ClearEngine` 是 numpy 精确算术加上严格的 level/scale
记账，所以下面的数字证明的是调度和代数正确，不含 CKKS 噪声。GPU FHE 路径会在此基础上叠加
CKKS 噪声，但 best-fit scale 那一列应该完全一致——哪一级 scale 先偏掉就是那一级的问题。

## 配置

| 参数 | 值 |
|------|-----|
| engine | clear |
| layers | 1 |
| samples | 4 (MRPC validation) |
| depth | 90 |
| log_n | 16 |
| scaling_bits | 50 |
| first_mod_bits | 60 |
| dnum | 3 |
| model | textattack/bert-base-uncased-MRPC |

## Accuracy (against dataset labels, 4 samples)

| 路径 | Accuracy | F1 | TP | FP | FN | TN |
|------|------:|------:|---:|---:|---:|---:|
| Plaintext | 100.00% | 100.00% | 2 | 0 | 0 | 2 |
| Encrypted | 100.00% | 100.00% | 2 | 0 | 0 | 2 |

## End-to-end fidelity (against plaintext model)

| 指标 | MAE | RMSE | Max Abs | relRMSE |
|------|------:|------:|------:|------:|
| Hidden (layer 0 output) | 4.437e-4 | 5.819e-4 | 3.828e-3 | 1.039e-3 |
| Hidden (rescaled by best-fit 1.9998) | 4.424e-4 | 5.803e-4 | 3.864e-3 | 1.036e-3 |
| Logits | 6.656e-4 | 9.608e-4 | 1.846e-3 | 5.302e-4 |
| Probabilities | L1 mean 3.571e-4 | L1 max 1.362e-3 | label agreement 100% | — |
| Best-fit scale | 1.9998 | | | |

## Per-stage fidelity (layer 0, sample 0, each rescaled by its best fit)

| Stage | Best-fit Scale | MAE | RMSE | Max Abs | relRMSE |
|-------|------:|------:|------:|------:|------:|
| query | 2.0000 | 1.888e-7 | 2.641e-7 | 2.536e-6 | 2.884e-7 |
| value | 2.0000 | 1.204e-7 | 1.673e-7 | 1.390e-6 | 3.141e-7 |
| scores (stage 06) | 0.5000 | 3.763e-7 | 5.061e-7 | 3.695e-6 | 2.871e-7 |
| softmax (stage 07) | 1.0023 | 7.132e-8 | 1.406e-7 | 3.227e-6 | 2.306e-6 |
| attention_dense (10) | 2.0044 | 5.566e-5 | 7.917e-5 | 7.037e-4 | 3.001e-4 |
| norm_1 (11) | 1.9998 | 6.214e-4 | 8.385e-4 | 1.163e-2 | 6.509e-4 |
| intermediate (12) | 0.0313 | 9.014e-4 | 1.176e-3 | 1.627e-2 | 4.776e-4 |
| gelu (13) | 2.0001 | 3.670e-4 | 5.401e-4 | 1.272e-2 | 2.283e-3 |
| output_dense (14) | 2.0000 | 9.990e-4 | 1.290e-3 | 1.539e-2 | 2.197e-3 |
| norm_2 (16) | 1.9998 | 5.852e-4 | 7.458e-4 | 3.702e-3 | 1.121e-3 |

## Timing

| Phase | Time | % |
|-------|------:|------:|
| layer 0 | 17.10s | 47.7% |
| encode weights | 13.52s | 37.7% |
| plaintext tail | 3.35s | 9.3% |
| plaintext forward | 1.07s | 3.0% |
| encrypt | 0.63s | 1.8% |
| decrypt | 0.20s | 0.6% |
| **total** | **35.87s** | **100%** |
| per sample | 8.97s | |

## 说明

- **softmax scale 1.0023**：不是误差，是 THOR 源码末尾 `int(1/(2*D_delta))+1` 的固定轻微过归一化。
- **query/value/scores** 的 relRMSE ~3e-7 是 float32 精度极限，说明 stage 01-06 代数完全正确。
- **gelu relRMSE 2.3e-3** 最大，因为 GELU 多项式拟合本身有近似误差（`[-1,1]` 上的 4 次多项式）。
- **层输出 relRMSE 1.04e-3** 与草稿 `thor-openfhe` v21 口径（RMSE ~1.3e-3、max ~9e-3）一致。
- **best-fit scale 1.9998**：层输出携带 2 倍值（THOR 不变量），rescale 后几乎精确。
