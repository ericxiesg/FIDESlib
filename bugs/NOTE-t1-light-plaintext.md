# T1 light plaintext —— 新功能，需要在真机上验证

> **2026-09-03 结项**：远程已在 GV100 上验证（见 `FIX-commit-6316173-bugs.md`）。
> 下面那个"最需要盯的一点：NTT 的输入次序"**推断正确**，不需要 bit reverse——stage5 的 8 个用例
> CPU/CUDA 全过。OpenFHE 拼写也都对。唯一遗留是 CPU 侧 `GetCKKSPackedValue()` 读的是
> `CKKSPackedEncoding` 的缓存值（换 element 不会刷新缓存），已把测试改成用乘法验证并写进
> `docs/light_plaintext.md`，THOR 路径不受影响。本文件其余内容保留作记录。

和 `RESPONSE-gpu-key-grow.md` 是同一批推送里的两件事，这一件是**新功能**不是 bug 修复。
同样，本机没有 GPU，**一行都没编译过**。设计和取舍写在 `docs/light_plaintext.md`，这里只讲怎么验。

## 是什么

THOR 的权重明文（BERT-base 约 22 万个）按常规编码是 `(L+1)` 个 RNS tower × N 个 64 位字，
N=2^16、L=30 时每个 16 MiB，加起来 ~110 GiB，V100 的 32 GB 一层都放不下。

但 CKKS 编码可以拆成"与 level 无关的部分"和"逐 level 的投影"：

```
message --特殊 IFFT--> 实数 --×Δ 取整--> N 个整数系数 --mod q_i, NTT--> towers
        \___________ 与 level 无关 ___________/          \__ 逐 level __/
```

FIXEDMANUAL 下 Δ 不随 level 变，所以那 **N 个 int64 系数就是整个明文**：0.5 MiB，用的时候按密文的
level 展开。这就是 desilofhe 的 `encode_to_light_plaintext` / `write_light_plaintext` /
`read_light_plaintext`（`THOR/src/thor/he.py` 里每个权重和 mask 都走这三个调用）。

## 新增了什么

| 位置 | 内容 |
|---|---|
| `api/LightPlaintext.{hpp,cpp}` | `LightPlaintextImpl`（coeffs / scale / slots / noise_scale_deg / level_hint / uid）+ 二进制文件读写 |
| `api/CryptoContext.*` | `MakeLightPlaintext`、`ExpandLightPlaintext`、`EvalMult(ct, light)`、`EvalAdd(ct, light)`、FIFO 展开缓存、`GetConsumedLevels` |
| `src/CKKS/ElemenwiseBatchKernels.*` | `expandCentredCoeffs_` 核：每个系数对每个 limb 做一次有符号取模 |
| `src/CKKS/{LimbPartition,RNSPoly,Plaintext}.*` | `loadCentredCoefficients` / `Plaintext::loadLight`：上传 N 个 int64 → 逐 limb 取模 → 正向 NTT |
| `python/` | `Engine.encode_to_light_plaintext / write_light_plaintext / read_light_plaintext / expand_light_plaintext`，`multiply`/`add` 直接吃 LightPlaintext |
| `python/tests/test_stage5_light_plaintext.py` | 8 个用例 |

编码**不自己实现 CKKS encoder**：调 OpenFHE 的 `MakeCKKSPackedPlaintext`，转到
`Format::COEFFICIENT`，把第 0 个 tower 做中心提升（centre-lift）取出来。这样"紧凑路径"和"常规路径"
的任何差异都只可能是真差异，而不是我们自己 IFFT 的舍入。

## 最需要盯的一点：NTT 的输入次序

`RNSPoly::loadCentredCoefficients` 把**自然次序**（非位反转）的系数喂给 FIDESlib 的正向 NTT。
理由：`openfhe-interface/RawCiphertext.cuh` 里 `REVERSE == false`，说明 FIDESlib 的 evaluation
布局和 OpenFHE 的一致；而 FIDESlib 内部 INTT/NTT 互为逆变换，所以它的 coefficient 域应该就是
OpenFHE 的自然系数次序。

**这个推断没有在硬件上验证过。** 如果它错了，症状非常明确：

> `test_stage5_light_plaintext.py::test_multiply_matches_dense_encoding`
> **CPU 过、CUDA 不过**（CPU 路径是让 OpenFHE 自己做 NTT，不受影响）。

真出现这个症状，改法是在 `RNSPoly::loadCentredCoefficients` 上传前对系数做一次
`bit_reverse_vector`（`openfhe-interface` 里已有这个函数）。**先试这个再怀疑别的。**

## 其它可能对不上的 OpenFHE 拼写

`ExpandLightPlaintext` 的 CPU 分支用到这几个，都是从树里已有代码或 `Ciphertext.cpp` 里那段
OpenFHE 参考注释抄的，但补丁版 1.5.1.1 可能有出入，编不过直接改：

- `lbcrypto::DCRTPoly::PolyType`（= NativePoly）
- `DCRTPoly(const std::vector<PolyType>&)` 构造函数
- `NativePoly::operator[]` 赋值（`tower[j] = NativeInteger(...)`）
- `PlaintextImpl::GetElement<DCRTPoly>()` 当左值用
- `MakeLightPlaintext` 里的 `t1.GetValues()[j].ConvertToInt()`（和 `GetRawArray` 同一写法）

## 验证顺序

```bash
export PYTHONPATH=$PWD/python
PYFIDESLIB_DEVICES=cpu    pytest python/tests/test_stage5_light_plaintext.py -x -v
PYFIDESLIB_DEVICES=cuda:0 pytest python/tests/test_stage5_light_plaintext.py -x -v
# 然后全量
PYFIDESLIB_DEVICES=cpu    pytest python/tests -x -v
PYFIDESLIB_DEVICES=cuda:0 pytest python/tests -x -v
```

几个用例的含义：

- `test_multiply_matches_dense_encoding`：`multiply(ct, light)` 和 `multiply(ct, ndarray)` 必须差
  < 1e-9（同一个 encoder、同一组系数，不该有肉眼可见的差）。**这是主判据。**
- `test_expands_at_the_ciphertext_level`：同一个 light 明文在 level 12 / 9 / 5 上都能用 —— 这正是
  THOR 复用 mask 的方式。
- `test_message_too_large_for_one_tower_is_rejected`：Δ=2^50、q0≈2^55，消息量级 2^12 时系数超过
  q0/2，没有单 tower 表示。要么我们的跨 tower 校验拦下，要么 OpenFHE 自己的 overflow 检查拦下，
  **不能**默默给出一个错的明文。如果这个用例因为"没抛异常"而失败，说明校验逻辑有问题，请报告。
- `test_cache_is_bounded`：展开结果按 (uid, level) FIFO 缓存，容量默认 64。

## 顺便量一下

真机上跑通之后，麻烦顺手记两个数进 report，后面 T2/T5 排显存要用：

1. `light.nbytes()` 与同一消息常规编码后 GPU 上的字节数之比（应该 ≈ L+1）。
2. `ExpandLightPlaintext` 在 GPU 上的耗时（N=2^16、level=30 那一档），和一次 `EvalMultPt` 比。
   如果展开明显比乘法还贵，`expandCentredCoeffs_` 里那个 64 位 `%` 就得换成 Barrett
   （`C_.prime_better_barret_mu` 里有现成常数）。

## 已知未做

- THOR `encode_weights.py` 的落盘目录结构还没搬（属于 T2）。
- 展开后的明文在缓存里会占 `(level+1)` 个 tower 的显存。流式扫权重的 stage 应该设
  `light_plaintext_cache_capacity = 0`。
