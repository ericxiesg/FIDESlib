# 请给 fhe 子命令加三个 scale 参数

日期：2026-09-17。

`bench magnitudes` 已有 `--residual-scale`、`--refresh-scale`、`--score-refresh-scale`，
但 `bench fhe` 的 `run_encrypted()`（`bench.py:348-354`）创建 `EncoderLayer` 和
`encode_layer` 时没传这三个参数。

要跑 GPU 端到端，需要：

1. `fhe` 子命令加三个 `add_argument`（和 `magnitudes` 一样）
2. `run_encrypted()` 里的 `EncoderLayer(...)` 和 `encode_layer(...)` 传上 `args.residual_scale`、`args.refresh_scale`、`args.score_refresh_scale`

确定的三参数：

```
--residual-scale 256
--refresh-scale 4
--score-refresh-scale 16
```

加完后我这边跑：
```
python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
  --refresh-after-dense --binary-rotations \
  --scaling-bits 59 --first-mod-bits 60 \
  --residual-scale 256 --refresh-scale 4 --score-refresh-scale 16
```
