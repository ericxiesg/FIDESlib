# noise_level 全程=1，断言未触发，组合测试 49 passed

日期：2026-09-18。针对 `c163240`。

---

## 0. 结论

`noise_level` 漂移假设**排除**。在 `-UNDEBUG` 重编译后，`he_inv` 全部 7 轮 × 2 次调用 = 124 条探针，**每一条都是 1**，`multNoRelin` 里的 `assert(NoiseLevel == 1)` 没有触发任何 abort。组合测试 `test_he_inv_primitives.py` 49 个用例全部通过。

---

## 1. noise_level 探针

在 `numeric.py` 的 `he_inv` 和 `_restore_magnitude` 中，在 THORFHE_DEBUG=1 下打印每一步的 `engine.noise_level(ct)`：

```
[noise_level] pre-iter  a=1  b=1
[noise_level] iter01  correction=1  a_pre=1  b_pre=1
[noise_level] iter01  a_post_times=1  b_post_times=1
[noise_level] iter01  a_post_restore=1  b_post_restore=1
[noise_level] iter02  correction=1  a_pre=1  b_pre=1
[noise_level] iter02  a_post_times=1  b_post_times=1
[noise_level] iter02  a_post_restore=1  b_post_restore=1
[noise_level] iter03  correction=1  a_pre=1  b_pre=1
[noise_level] iter03  a_post_times=1  b_post_times=1
[noise_level] restore conjugate-add  pre=1  post=1
[noise_level] restore conjugate-add  pre=1  post=1
[noise_level] restore int-mult factor=15  a: 1->1  b: 1->1
[noise_level] iter03  a_post_restore=1  b_post_restore=1
...（一直到 iter07，全部 =1）
```

**统计**：124 条 noise_level 行，a 和 b 出现的值全部是 1，没有一条非 1。

### 构建确认

```
CMAKE_BUILD_TYPE:STRING=Release
CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG -UNDEBUG
```

`-UNDEBUG` 覆盖了 `-DNDEBUG`，`assert` 宏恢复生效。运行中没有 crash/abort/SIGABRT。

### bootstrap 的 noise_level

`Engine.bootstrap` 里有一个循环把 bootstrap 输出 rescale 回 1。探针证实 `pre-iter a=1 b=1`——bootstrap 出来确实被修到了 1，之后整个迭代维持 1。

---

## 2. 组合测试

`test_he_inv_primitives.py`（`a8a3719`），在 GPU 上跑：

```
PYFIDESLIB_DEVICES=cuda:0 python -m pytest tests/test_he_inv_primitives.py -q -s
49 passed in 1.83s
```

包括：
- `test_scalar_minus_ciphertext`：5 scalars × 4 drops = 20 用例
- `test_integer_scalar_is_exact_and_level_free`：6 factors × 4 drops = 24 用例
- `test_doubling_through_the_conjugate`：4 drops = 4 用例
- `test_a_goldschmidt_step_does_not_compound`：1 用例（5 步 Goldschmidt 组合）

**全部通过**。在小参数下组合也不发散。

注意：小参数用的是 `log_n=13, depth=12`，而真实跑是 `depth=37`。组合测试在低 level（drop 10）也通过了，但 depth 12 的 level 2 和 depth 37 的 level 2 可能有不同的模数链行为。

---

## 3. iter03_b 发散完全复现

探针结果和之前完全一致：

| step | a max | b max | 说明 |
|------|-------|-------|------|
| iter01 | 1.036 | 1.493 | OK |
| iter02 | 0.305 | 0.163 | OK |
| iter03 | 0.265 | **241.8** | ❌ b 首发发散 |
| iter04 | 8.87e+06 | 2.42e+07 | 💥 |
| iter05 | 5.93e+16 | 1.62e+17 | 💥 |

**第二个 he_inv 调用**（可能是 he_invsqrt 内部）从 iter01 就全坏了（max 8.28e+06），说明第一个 he_inv 的坏结果通过 `07d.inverse_denominator` 传导到了下游。

---

## 4. 排除表更新

```
✅ variance 灾难性相消        ✅ 自举越界
✅ ones 空槽泄漏              ✅ 分母空槽泄漏
✅ 承载槽相对噪声（线性）      ✅ correction 绝对误差（线性）
✅ 别名（const + 深拷贝）      ✅ prepare_for_multiply（恒等）
✅ 旋转密钥计划               ✅ 三个算子孤立测试
✅ noise_level 漂移           ✅ 组合测试（小参数）
✅ assert 禁用（-UNDEBUG 跑过）
```

**所有单点假设已穷尽。** 每个算子干净、noise_level 干净、别名干净、组合在小参数下干净——但 depth 37 的真实跑 iter03_b 就是 241.8。

---

## 5. 下一步方向

剩下的可能是**深度相关的**行为：

1. **低 level 下的 multiply 精度**：iter03 跑在 level 17，模数已经较小。组合测试在 depth 12 drop 10（level 2）通过了，但 depth 37 的 level 17 的模数大小可能不同。值得在**真实 depth 37 的参数下**跑 `test_a_goldschmidt_step_does_not_compound`。

2. **subtract 的 scalar-ciphertext 路径在低 level**：`correction = subtract(2/k * b.delta, b.ciphertext)` 是标量减密文。这个路径在 GPU 上的实现是否在低 level时有精度问题？组合测试的 `test_scalar_minus_ciphertext` 在 depth 12 下通过了，但真实 depth 37 下未测。

3. **rescale 在 FIXEDMANUAL 下的舍入**：`_times` 里 `rescale(relinearize(multiply(x,y)))`——rescale 在低 level 时的 RNS 舍入是否引入了少数 slot 的大误差？这在小参数下测不出来，因为模数链不同。

4. **decrypt 并对比单步**：在 iter03 的 correction 计算后 decrypt 它，和 ClearEngine 的同一步对比。如果 correction 本身就错了（不是后续 multiply 的问题），那问题在 `subtract` 或 `prepare_for_multiply` 在 depth 37 下的行为。

报告文件：`/home/zhiyuan/bench-run/gpu_noise_level.log`
探针修改：`python/thorfhe/numeric.py`（未 commit，是 debug 探针）

Co-Authored-By: CodeAgent <noreply@anthropic.com>
