# 回复 GPU-illegal-access-LTdotProduct-20260909

日期：2026-09-09。本机没有 GPU，**C++ 改动一行都没编译过**。

`CUDA_LAUNCH_BLOCKING=1` 这一步很值——把异步错误钉到了真正的 kernel 上，我上一轮猜的
`multMonomial` 是错的。而且你的两条推断（1「limb 越界」和 2「明文和密文 level 不匹配」）
基本就是答案。另外那 15 把键要 grow 的事，**我上一轮的修复没生效，原因找到了，是另一处**。

---

## 一、`LTdotProductPtBatch`：kernel 用 `out[0]` 的 level 定网格，却索引三个数组的 limb

```cpp
int limbsize = out[0]->getLimbSize(*out[0]->level);
...
grid = { cc.N / 64, (uint32_t)limbsize, 1u };      // ← grid.y 来自 out[0] 的 level
```

kernel 内部用 `blockIdx.y` 去索引 **out、in、pt 三个数组每一个 partition 的 `limb[]`**。
只要其中任何一个持有的 limb 数少于 `limbsize`，就会拿到越界指针 → 非法访存 → 在下一个同步点报出来。

**`pt` 是最可能对不上的那个**：自举的 StC/CtS 明文是**按自己的 level 预计算**的，
和密文当前 level 不是同一个东西。这正是你的推断 2。

已按你的建议 1 加了检查（三个数组都查），并且会**指名道姓**：

```
FIDESlib: LTdotProductPtBatch would read 29 limbs from pt[7], which holds 24.
The operands are at different levels.
```

下一次跑就能直接知道是 `out` / `in` / 还是 `pt` 对不上、差多少。**这比继续猜快得多**——
如果是 `pt`，那就是明文预计算的 level 和密文 level 的对齐问题；如果是 `in`，那是辅助 poly 的
level/limb 不一致（我上一轮在 `multMonomial` 里加的那个检查是同一类问题）。

## 二、那 15 把键要 grow：我上一轮改错了地方

上一轮我改的是 `ContextData::AddRotationKey`（`std::map::emplace` 静默丢弃）。方向对，
但**根本轮不到它执行**：

```cpp
for (int i : indexes2) {
    if (i && !GPUcc->HasRotationKey(i, publicKey->GetKeyTag())) {   // ← 这里
        indexes3.emplace_back(i);
    }
}
```

`AddRotationKeys` **对已经存在键的索引直接跳过，连生成都不生成**。所以：

1. 第一次调用（你的电路计划）加进截断到 28 的键；
2. 第二次调用（自举计划）遇到同一个索引 → `HasRotationKey` 为真 → **整个跳过**；
3. 我那个「重复时保留覆盖更高 level 的」逻辑在 `AddRotationKey` 里，**永远没被调用到**。

已改成：**只有当已有的键确实覆盖了本次计划的需求才跳过**，否则重新生成；
`AddRotationKey` 那一层再保留更大的那把。两处合起来才是完整的修复。

顺带修了一个会立刻踩到的坑：`HasRotationKey` 和 `AddRotationKey` 会把负索引归一化，
**`GetRotationKey` 不会**，而自举索引是带符号的——所以查询前统一归一化了一次。

**所以请去掉「所有键设成 level 34」那个 workaround 再跑**，正常的截断计划这次应该能用了。
`--allow-key-grow` 下的 "grown at runtime" 应该变成 **0**；如果还不是 0，把还在 grow 的索引和
它们的 (计划 level, 实际 level) 发回来，那就说明是 `GetBootstrapKeyLevelPlan` 的逐层模型有偏，
我按那组数改。

## 三、关于 compute-sanitizer 跑不动

sanitizer 要额外约 2 GiB。可以这么腾：

```bash
# 只建 context 不跑推理，先确认 keygen 干净
compute-sanitizer --tool memcheck python -m thorfhe.bench budget ...   # 不需要 GPU

# 或者把工作集压到最小再跑
compute-sanitizer --tool memcheck --force-blocking-launches \
  python -m thorfhe.bench fhe --engine fideslib --layers 1 --limit 1 \
      --binary-rotations --refresh-after-dense --depth 34 --dnum 4 \
      --light-plaintext-cache 1 --card-gib 30
```

不过**有了第一节那个检查，多半用不着 sanitizer 了**——如果确实是 limb 数对不上，
异常信息会直接说是哪个数组、差多少。先跑普通模式看有没有抛出那句话。

---

## 四、下一次运行建议（每步都能单独给结论）

1. **拉这版，去掉 workaround，正常跑一次。** 期待三种结果之一：
   - 抛出第一节那句 `LTdotProductPtBatch would read N limbs from pt[k]...` → 我们拿到了确切原因；
   - 抛出 `multMonomial found N limbs where...` → 是另一处，同一类问题；
   - 跑过去了 → 第二节那个键的修复本身就是原因（截断键被自举误用，破坏了明文/密文对齐）。
2. **带 `--allow-key-grow` 再跑一次**，确认 "grown at runtime" 是 0。
3. 两步都过了，就是完整的一层了。

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `src/CKKS/LimbPartitionBatch.cu` | `LTdotProductPtBatch` 对 out/in/pt 三个数组查 limb 数，不足则抛出带数组名和数字的异常 **未编译** |
| `src/CKKS/openfhe-interface/RawCiphertext.cu` | `AddRotationKeys` 不再「有键就跳过」，改为「已有的键覆盖不够就重新生成」；查询前归一化索引 **未编译** |
