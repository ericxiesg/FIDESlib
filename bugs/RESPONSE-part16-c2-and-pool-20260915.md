# 回复 Part16：三条核心论断我全核了，都成立；前两条已改

日期：2026-09-15。**C++ 未编译**（本机无 GPU）。

这份报告的判断我认同，尤其是开头那句——**"噪声爆炸"是误诊**。
下面按它的排查顺序回，能立刻做的都做了。

---

## 一、核实结果

### 1.1 "1e124 不可能是噪声" —— 成立，而且我们自己的数据早就指向它

`Q_l/Δ` 在 9 limb 那一档是 8.3e121，观测 1e124 落在同一档。
**噪声只能把结果推向 0，推不到 1e124。** 这条判据我们之前没用上，是我们的疏忽——
上一轮我还在跟远程争论"bootstrap 是重置噪声不是累积"，
但我没往前再走一步说"所以这根本不是噪声问题"。**报告走到了，我没走到。**

### 1.2 `c2` 被静默丢弃 —— 成立，逐个函数查过

```
Ciphertext::rotate            无 c2
Ciphertext::conjugate         无 c2
Ciphertext::multMonomial      无 c2
Ciphertext::rotate_hoisted    无 c2
multIntScalar                 无 c2
```

**五个全中，而且一句 `assert(!c2)` 都没有。** 我在 2026-09-11 的
`SWEEP-changes-that-affect-other-workloads` 里写过一句"`multIntScalar` 不处理 c2，
等惰性重线性化用起来再补，现在没有证据说明被触发"——**那句话是错的**：
`evaluate_polynomial` 从一开始就在用 c2，证据一直都在。

### 1.3 `c2` 没有复位回收池 —— 成立，这是最具体的一条

```cpp
// 构造函数（上游）                      // 我们的 c2 路径
c0(cc->getAuxilarPoly()),                c2 = make_unique<RNSPoly>(cc.getAuxilarPoly());
c0.dropToLevel(-1);   ← 复位             c2->SetModUp(false);       ← 没有
c1.dropToLevel(-1);                      c2->grow(c1.getLevel());   ← grow 会早退
```

四个申请点（`:207`、`:257`、`:763`、`:1466`）**全都没有复位**。
而且报告指出的"短测试池是空的所以全对、跑满一层池被填满才出错"——
**这正好解释了为什么 98 个单测全绿而 benchmark 炸**，我们一直没解释这一点。

---

## 二、已经改的（你的第 1、2 条）

### 2.1 五个算子加 `requireDegreeOne` throw

```cpp
void Ciphertext::requireDegreeOne(const char* what) const {
    if (c2)
        throw std::runtime_error(std::string("FIDESlib: ") + what +
            " on a degree-2 ciphertext. Call relinearize() first: this operation carries c0 and c1 "
            "only and would discard the third component, leaving a ciphertext that decrypts to an "
            "unrelated value rather than failing.");
}
```

加在 `rotate`、`conjugate`、`multMonomial`、`rotate_hoisted`、`multIntScalar`。
**用 throw 不用 assert**——正如你指出的，Release 下 assert 会被编译掉，
而产出 1e124 的那次跑的就是 Release。

### 2.2 四个 c2 申请点补 `dropToLevel(-1)`

照抄构造函数的写法，四处全补。**这一条是真修 bug，不只是加诊断。**

### 2.3 顺带修了你点出的两处不一致

* **`bench.py --depth` 默认 90 → 37**，并在 help 里写明"可用点是 37 配 `--refresh-after-dense`"。
  （`--refresh-after-dense` 的默认值我**没动**——翻它会影响远程在跑的实验，请你们定。）
* **`secret_key_dist` 不一致，而且是在我自己的测试里**：
  `test_polynomial_reproducer.py` 和 `test_bootstrap_noise_level.py` 的"基准参数"
  漏了这个字段，于是用 `UNIFORM_TERNARY` 建引擎去测一个跑 `SPARSE_TERNARY` 的配置——
  按你 5.2 的说法，那等于 bootstrap 用了另一套 Chebyshev 系数。已改成显式传 `SPARSE_TERNARY`，
  并做成惰性（否则没编扩展的机器上整个模块收集失败）。

98 个用例仍全过。

---

## 三、我的判断和你略有不同的地方

### 3.1 第 2 节和第 3 节是**同一个 bug 的两面**，不是两个嫌疑

`c2` 从池里拿到一个 level 不是 -1 的多项式 → `grow()` 早退不分配 →
`limb.size()` 比 level 声称的少 → 后面任何按 level 索引 limb 的算子读到野指针。
**而"c2 被丢弃"和"c2 的 limb 不够"会产生同一种症状**（结果与真值无关）。

所以 2.1 和 2.2 两个改动要**一起**编译验证：如果跑完不抛异常且结果正常，
是 2.2 修好的；如果抛异常，栈顶告诉我们是 2.1 命中的。**两者都不命中才说明还有第三个原因。**

### 3.2 参数那一条我同意方向，但顺序建议往后放

Δ 从 2^59 掉到 2^50 少 9 bit 是事实，EasyFHE 那组参数被验证过也是事实。
但**在 c2 的问题修掉之前换参数，会把一个结构性 bug 和一个精度问题混在一起**——
这正是我们这个项目已经犯过三次的错（`EvalNegate`、噪声底噪、bootstrap，三次都猜错方向）。

**建议顺序**：先编译验证 2.1+2.2 → 再跑 `test_polynomial_reproducer.py`（你的第 6 条，我同意"先跑它别跑 benchmark"）→
**然后**才动参数。

### 3.3 关于 `--depth` 默认 90 那条：失败那次应该不是它

远程一直显式传 `--depth 37 --refresh-after-dense`（历次报告的复现命令里都有），
所以默认值虽然确实是个陷阱、也确实该修，但**不是失败那次的原因**。

---

## 四、你要的调试信息，我能先给的

你第 7 节列的清单里，有几项我这边就能答：

* **编译类型**：`Release`（`-DCMAKE_BUILD_TYPE=Release`，见历次复现步骤）——
  所以你说的"assert 全部失效"是对的，这也是我把守卫写成 throw 的原因。
* **`--depth` 等实际参数**：`--depth 37 --dnum 4 --bootstrap-level-budget 3,3
  --binary-rotations --refresh-after-dense --light-plaintext-cache 4 --per-stage --device-memory`。
* **`binary_rotations` 是开着的**（你 5.3 的担心成立）：远程每次都带 `--binary-rotations`。
  按 `docs/thor_port.md` 实测是 8138 次 key-switch/层 vs 1802 次。
  **这确实是 ClearEngine 完全看不见的噪声**，值得在参数那一轮一起评估。
* **`test_stage4_bootstrap.py` 跑在 59/60、log_n 13、depth 25** —— 你说"等于没测"，属实。

其余几项（probe 完整输出、设备上的 reproducer 结果、`[FIDESlib]` 日志）要等远程跑。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CKKS/Ciphertext.{cpp,cuh}` | `requireDegreeOne`；`rotate`/`conjugate`/`multMonomial`/`rotate_hoisted` 加守卫；四个 c2 申请点补 `dropToLevel(-1)` **未编译** |
| `src/CKKS/ApproxModEval.cu` | `multIntScalar` 加守卫 **未编译** |
| `python/thorfhe/bench.py` | `--depth` 默认 90 → 37，help 写明配套 flag |
| `python/tests/test_polynomial_reproducer.py`、`test_bootstrap_noise_level.py` | 基准参数补 `secret_key_dist=SPARSE_TERNARY`，改为惰性求值 |
