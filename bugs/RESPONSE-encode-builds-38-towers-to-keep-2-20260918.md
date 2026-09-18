# 找到了：`MakeLightPlaintext` 建 38 个 tower，只读其中 2 个

日期：2026-09-18。承接 `a56aad6`。**这份带一个待编译验证的 C++ 改动。**

---

## 0. 结论

`api/CryptoContext.cpp:2186`（改之前）：

```cpp
// Level 0 keeps every tower, which is what the single-tower cross-check below wants.
const uint32_t encodeLevel = levelHint >= 0 ? static_cast<uint32_t>(levelHint) : 0u;
lbcrypto::Plaintext pt = context->MakeCKKSPackedPlaintext(value, 1, encodeLevel, nullptr, slots);
```

`Stages.plaintext` 调 `encode_to_light_plaintext(value)` **不带 level** → `levelHint = -1`
→ `encodeLevel = 0` → OpenFHE 建**整条模数链**。depth 37 就是 **38 个 tower，38 次 N=65536 的 NTT**。

然后下面只读两个：

- `t0` = `poly.GetElementAtIndex(0)` —— 取系数
- `t1` = `poly.GetElementAtIndex(1)` —— 单 tower 越界检查

**36 个 tower 算完就扔。** 58 ms / 38 ≈ 1.5 ms 一次 NTT，量级完全对得上。

---

## 1. 为什么可以只建 2 个

`docs/light_plaintext.md` 自己写的：

```
message --special IFFT--> reals --*Delta, round--> N integer coefficients --mod q_i, NTT--> towers
        \____________________ level independent ____________________/      \__ per level __/
```

> Under FIXEDMANUAL (and FIXEDAUTO) the scaling factor `Delta` does not depend on the level, so the
> integer coefficient vector *is* the plaintext.

**系数按设计就与 level 无关**——这正是 light plaintext 存在的理由。所以在哪个 level 编码，
出来的 `coeffs` 逐位相同；level 只决定 OpenFHE 白算多少个 tower。

FLEXIBLE* 下 `Delta` 确实随 level 变，所以那条路仍然走 `levelHint`（和
`ExpandLightPlaintext` 里已有的那个 guard 一致）。

---

## 2. 改动（已推，**我这边编译不了，请你构建验证**）

```cpp
const auto encodeParams       = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(context->GetCryptoParameters());
const auto encodeTechnique    = encodeParams->GetScalingTechnique();
const bool encodeLevelMatters = encodeTechnique == lbcrypto::FLEXIBLEAUTO || encodeTechnique == lbcrypto::FLEXIBLEAUTOEXT;
const size_t towers           = encodeParams->GetElementParams()->GetParams().size();
// two towers left: the value, and the one the check below compares it against
const uint32_t cheapestLevel  = towers > 2 ? static_cast<uint32_t>(towers - 2) : 0u;
const uint32_t encodeLevel =
    encodeLevelMatters ? (levelHint >= 0 ? static_cast<uint32_t>(levelHint) : 0u) : cheapestLevel;
```

预期：38 tower → 2，**约 19 倍**。

| | 现在 | 预期 |
|---|---|---|
| encode_to_light_plaintext | 58 ms × 19,137 = 1109 s | ~3 ms × 19,137 = **~58 s** |
| 一层总计 | 1154 s | **~100 s** |

### 请验证三件事

1. **编译过**（我没有 CUDA，只能静态改）。`GetElementParams()` / `GetScalingTechnique()`
   的拼写按 `ExpandLightPlaintext` 里现成的用法抄的，但没编译过。
2. **精度逐位不变**。`--per-stage` 的 `relRMSE` 应该和改之前**完全一样**——
   如果变了，说明系数不是 level 无关的，那这个改动要退回，而且 light plaintext 的前提有问题。
3. **`--time-ops` 再跑一次**，看 `encode_to_light_plaintext` 掉到多少。

如果 `MakeCKKSPackedPlaintext` 在某个 level 上抛"level exceeds depth"之类的，
把 `towers - 2` 调成 `towers - 3` 试试；边界我只能靠你那边的报错定。

---

## 3. 你报告里两处我想修正

### 3.1 不是"惰性编码"的锅

> `--compact` 模式本应"惰性权重编码"——但惰性编码发生在计算过程中，不是预编码。

`--lazy-weights` 改变的是**权重 numpy 数组**什么时候构造（省宿主内存，9.7 GiB → 3.2 GiB），
不改变 `encode_to_light_plaintext` 调多少次。开不开 `--compact`，编码次数都是一样的。
真正的问题是**每次编码贵了 19 倍**，不是编码发生的时机。

### 3.2 预编码（你的方案 A）救不了

你自己也注意到了：「预编码本身也要 1110 s」。把同样的 19,137 次编码挪到前面，
总时间不变——除非放到**另一个线程**和 GPU 重叠，而 GPU 那边只有 44 秒的活，
重叠最多省 44 秒。

**先修单次编码的成本**，这是 19 倍；之后再谈调度。

### 3.3 但方案 C 值得做，而且比你说的大

> C. 缓存 encode 结果……但 THOR 一层每个权重只用一次。

一层之内是这样。但：

- **12 层 × 每个 sample** 都要重编码同一批权重。权重不随 sample 变。
- `weights_io.py` 已经有 `write_light_plaintext` / `read_light_plaintext`，
  `STATUS.md:149` 也记着"需要缓存到磁盘"。

编一次存盘，之后每次跑都省掉全部编码时间。和 §2 的改动叠加：
第一次 ~58 s/层，之后 **0**。

---

## 4. 我这边的 21,195 vs 你的 19,137

我在本机数过一层带 ndarray 的 multiply 是 **21,195** 次，而 `Stages.plaintext`
的内容缓存只认得 **676** 个不同内容——但那是**干运行**，权重是全零，所以全都哈希成同一个。
真实权重下每个权重矩阵都不同，缓存基本不命中，于是 21,195 次里有 19,137 次真的去编码。
两个数字是一致的，我之前引用 676 是错的上下文。

顺带：`Stages.plaintext` 为了查这张表，每层要 `tobytes()` + hash **9.69 GiB**（102 µs/次）。
在 1109 s 面前那只有 2.2 s，等 §2 落地之后再看要不要动。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
