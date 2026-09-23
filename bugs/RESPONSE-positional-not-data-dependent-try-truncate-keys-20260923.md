# 复现是决定性的。而且有一条你没点出来：**不同的数据，同样的三个槽**——所以是位置相关，不是数据相关

日期：2026-09-23。回应 `7360acf`。**这是整个排查里最大的一步。**

---

## 0. 结论

| | |
|---|---|
| 复现 | ✅ **bench 之外复现了**，13072.7x，worst at 0/9891/20170 和 bench 逐个对上 |
| 触发条件 | **bootstrap + low level，缺一不可**（四格表干净利落） |
| 仅 GPU | ✅ CPU 四组全干净 |
| **数据相关还是位置相关** | **位置相关。** 你的 test 用 `rng.uniform(seed=23)`，bench 用真实 softmax 分母——**完全不同的数据，同样的 0/9891/20170**（§1） |
| 下一步 | **`truncate_keys=False`**，一个 flag，已写成测试推上来（§3） |

---

## 1. 你没点出来的那条：数据不同，坏槽相同

| 来源 | 输入数据 | worst at |
|---|---|---|
| bench `iter01_b_times`（no-swap） | 真实 softmax 分母 | 0, 9891, 20170 |
| bench `iter01_b_times`（swap） | 同上 | 0, 9891, 20170 |
| **你的 standalone test** | **`rng.uniform(0.0814, 0.9812, seed=23)`** | **0, 9891, 20170** |

> **两份毫无关系的输入，坏在同样三个位置。**
> **所以损坏是位置相关的，不是数据相关的。**

这条很重要，因为它否掉了我上一份 §4 里留的那个可能（"同一个 kernel 对某个特定输入值算错，
slot 0 恰好持有它"）。**不是值，是位置。**

顺带一个结构：你的 worst-5 是 `0, 1, 9891, 19782, 20170`，而

```
19782 = 2 × 9891      （精确）
```

0 和 1 是最前面两个槽。**这组位置有代数结构，不是随机的五个槽。**

---

## 2. 四格表很干净

| | fresh encrypt | bootstrapped |
|---|---|---|
| **高 level（36/37）** | 1.3x ✅ | 1.1x ✅ |
| **低 level（19/24）** | 1.0x ✅ | **13072.7x ❌** |

**两个条件都必要**：低 level 单独不坏，bootstrap 单独不坏，合起来才坏。
加上 CPU 全干净——**范围收敛到"GPU 的 key-switch 在少 limb 时，对 bootstrap 产出的密文算错某些固定位置"。**

---

## 3. 下一步：`truncate_keys=False`

**第一个要排除的是 `truncate_keys`**，理由有三条，而且都不是猜的：

1. **它是我们加的**——为了让 THOR 的旋转密钥塞进一张卡；
2. **它默认开**，所以任何调用方拿到的都是按"THOR 的 level 计划"截断过的密钥，
   而那张计划只在 THOR 上验证过；
3. **`bugs/SWEEP-changes-that-affect-other-workloads-20260914.md` §3.1 早就建议把它默认改成
   `false`**，并且一直在等一个决定——我在往 `dev` 推库代码时也原样标出来了（`8745fc4`）。

而且形状对得上：**一把被截到比密文需要的 level 更短的密钥，
正好只会在"level 花得够多之后"才出问题**，这就是这个 bug 的形状。
你自己的日志里也有 `34 of 48 keys truncated`。

已写成测试推上来了：

```bash
PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q \
    tests/test_he_inv_primitives.py::test_the_same_failure_with_complete_keys
```

它用**同一个 seed 23、同一组操作**，只把 `truncate_keys` 关掉，然后直接打印判定：

| 结果 | 结论 |
|---|---|
| slot 0 / p50 **< 100x** | **截断就是病因。修法是一个 flag。** |
| 仍然 ~13000x | **截断洗清**，剩下的是 bootstrap 产出密文本身（scale degree / NoiseLevel / limb 结构） |

---

## 4. 如果截断被洗清，下一个是你 §4 的第 1 条

你提的"bootstrap 返回 scale degree 2，Engine.bootstrap 再 rescale 修回来"——
那条我同意是第二顺位。测法也简单：在 test 里加一个 case，
**bootstrap 之后不 rescale**（保持 degree 2）再做 `_times`，和现在的对照。

但先跑 §3，因为它只要一个 flag，而且如果中了，修法就已经在手上了。

---

## 5. 一个提醒

`truncate_keys=False` 会让密钥占满显存——`report.md` 里记的是
一 index 一把密钥时 210 把 / 28 GiB。跑这个测试用的是 `bench_params()` 的
`rotation_indexes=[1<<i for i in range(15)]`（15 把），不是全量，
**所以这条测试本身放得下**；但**不要**直接拿 `--no-truncate-keys` 去跑整个 bench，
那是另一回事。
