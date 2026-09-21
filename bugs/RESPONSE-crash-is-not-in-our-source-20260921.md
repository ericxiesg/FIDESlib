# 崩溃不在我们的源码里：两个 C++ commit 都能排除，嫌疑是**构建环境**

日期：2026-09-21。针对 `c355b11`。

---

## 0. 直接回答你的问题 1 和 2

### Q1：`7a03a0d` 的 2-tower 编码影响 bootstrap 对角线的 level 吗？**不可能。**

`MakeLightPlaintext` 在整个仓库里**只有两个调用点**：

```
python/src/bindings.cpp:166     ← Python 绑定
api/CryptoContext.cpp:2245      ← 它自己的 double 重载
```

**`src/` 里没有任何地方调它。** 而 bootstrap 的预计算是
`FIDESlib::CKKS::AddBootstrapPrecomputation`，在 `src/CKKS/openfhe-interface/RawCiphertext.cu`，
和 `MakeLightPlaintext` 没有任何关系。

我那个改动整个包在 `MakeLightPlaintext` 的函数体里，`cheapestLevel` 是个局部变量。
**它够不着 StC 对角线。**

### Q2：`fb5d499` 的 `stopAfterStage` 呢？**也不可能。**

它只加了四个提前返回，每个都是 `if (stopAfterStage == N) return;`。而调用方：

```
api/CryptoContext.cpp:1803   Bootstrap(*res_gpu, res_gpu->slots, prescaled);   ← 不传，默认 -1
pyfideslib/__init__.py:230   self.cc.EvalBootstrap(x)                          ← 不传，默认 -1
```

只有 `bootstrap_stage()`（诊断用，benchmark 从不调）会传别的值。
**`-1` 时四个条件全不成立，执行路径和改之前逐条相同。**

---

## 1. 那一层 level 是在哪丢的

```
CtS:  pts=36  ctxt=36   ✅
StC:  pts=35  ctxt=34   ❌
```

两次 LinearTransform 之间只有一件事：**近似模约减**（sine + double-angle），
也就是 `ApproxModEval.cu`。密文比预计算多掉了一层，说明那里多做了一次 rescale。

`8fee326..a49bae5` 的 diff **没有碰 `ApproxModEval.cu`**，也没碰 `CoeffsToSlots.cu`、
`LinearTransform.cu`、`Bootstrap.cu` 的非 `stopAfterStage` 部分。

**所以变的不是源码。**

---

## 2. 你自己记下的那条环境变化，我认为就是它

你附录里写着：

```
CUDA: /usr/local/cuda → cuda-13.3（但 cmake cache 里 CUDA_PATH=cuda-12.9）
```

- 能跑的 install：**Sep 16** 构建
- 崩溃的 install：**Sep 21** 构建

**工具链在这中间换了。** 这同时解释了你最困惑的那件事——
"回退源码到 `8fee326` 仍然崩溃"。因为问题本来就不在源码里。

### 还有一个更难受的可能：**混合的 .a**

你说 cmake 没追踪 header 变化，回退源码后 `.o` 不重编译。那反过来也成立：
你 Sep 21 install 的那个 `build/fideslib.a`，是在一棵**源码变过而 .o 没全部重编**的树上
增量构建出来的。

`fb5d499` 改了 `Bootstrap.cuh` 的函数签名（加了一个参数）。如果一部分 `.o` 是对着
**3 参数**的头文件编的、另一部分对着 **4 参数**的——那就是 ODR 违反，
inline 函数和常量在不同 translation unit 里可以是不同的东西，
**表现就是这种"只差一点点"的静默错误**。

---

## 3. 请按这个顺序做（第 1 步不是可选的）

### 1. 全量干净重编译，CUDA 版本钉死

```bash
rm -rf build build-py fideslib-install
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc   # 和 Sep 16 那次一致
cmake --build build -j && cmake --install build
# 然后重编 .so
```

5 分钟，但**在这之前任何"回退 X 仍然崩溃"的结论都不可靠**——包括你已经做的那三个。

- **不崩了** → 是工具链或混合 .o，和我们的源码无关，继续往下跑。
- **还崩** → 才轮到源码，去做第 2 步。

### 2. 如果还崩：GIL 现在是个编译开关（本次已推）

你要的"有 `stopAfterStage` 但没 GIL release"的 bindings.cpp，**不用手工合成了**：

```bash
cmake -S python -B build-py -DTHORFHE_RELEASE_GIL=0     # 或加进 CXX flags
```

`bindings.cpp` 里 25 处 `py::call_guard<py::gil_scoped_release>()` 全部换成了
`THORFHE_GIL` 宏，默认 1（释放），`-DTHORFHE_RELEASE_GIL=0` 就是 `4c98d0b` 之前的行为。
两种展开我都用预处理器验过语法。只影响 .so，不用重编 fideslib.a。

### 3. 还崩的话，告诉我 `ApproxModEval` 里 rescale 的次数

崩溃点已经缩到 CtS 和 StC 之间那一段。如果源码没变而 level 变了，
我想看 `ApproxModEval.cu` 里那几个**条件 rescale** 的条件在新旧编译器下是不是算出了不同结果
（`-ffast-math` 类的差异，或者 double 精度）。

---

## 4. 另外两件事

**你的 `--inverse-lift 3` 结果是有效的，而且是进展。** `07d` 从 **5.3e16 → 7.33e05**，
11 个数量级。它跑在旧 install 上，而旧 install 正是"没坏"的那个，所以这个数**站得住**。
（还不对——应该是 0.15 量级——但 padding floor + lift 两条确实在起作用。）

**另外提醒**：等你重新跑起来，拉了 `42a43ed`（每层方差窗口）之后
**level 消耗变了，旋转密钥必须重新生成**，不能复用旧 key cache。
`--plaintext-cache` 不受影响（明文编码和 level 无关）。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
