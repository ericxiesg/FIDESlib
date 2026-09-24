# StC 那个 modulus mismatch:免费就能定位到哪一半

2026-09-24。补 `e96b5b0` 第 3 节。

## 1. 零成本定位

`SlotsToCoeffs` 测试的 stdout 顺序是固定的:

```
Run SlotsToCoeffs            <- 之后是 CPU 参照:FHE->EvalSlotsToCoeffs + RescaleInPlace + Decrypt
After SlotsToCoeffs:         <- CPU 这一半活过来了
Result <...>
Run SlotsToCoeffs GPU        <- 之后是 GPU:EvalCoeffsToSlots(.., true) + store + GetOpenFHECipherText + Decrypt
Batch 100
```

**崩溃前最后打印的那一行,直接决定是哪一半。** 请把它贴出来——不用改任何代码,不用重跑别的东西。

- 停在 `Run SlotsToCoeffs` 之后 → **CPU 侧**,即 OpenFHE 自己的 `EvalSlotsToCoeffs` 或紧跟的 `RescaleInPlace` 在这组参数下就不成立。那样的话问题在 OpenFHE 的 sparse 预计算,和 GPU 无关。
- 停在 `Run SlotsToCoeffs GPU` 之后 → **GPU 侧**,即 GPU 结果搬回 OpenFHE 之后 `Decrypt` 里的 `c1 * s` 撞上模数不一致。那才是 `e96b5b0` §3 说的那类缺陷。

这两种情况的后续完全不同,而区分它们只需要一行日志。

## 2. 我查了一个假设,它是错的,但顺带查出一个真隐患

我本来怀疑:`GetOpenFHECipherText` 在 `RawCiphertext.cu:103-104` 做 `dcrt_0.resize(raw.numRes)`,如果 GPU 密文的 residue 数**多于**模板密文,`resize` 会用**默认构造、没有模数**的 Poly 把向量撑大,`Decrypt` 的 `c1 * s` 就正好抛 `Modulus mismatch`;而 `:89` 那个 `assert(size >= raw.numRes)` 在 release 下被编掉了。

**这个假设不成立。** `:84-85` 有个钳位:

```cpp
if (size < raw.numRes) {
    raw.numRes = size;
}
```

所以 `raw.numRes` 永远不会超过模板的 size,向量只会被截短、不会被撑大,`:89` 的 assert 也因此永远不会触发。崩溃不是这么来的。

**但这个钳位本身是个隐患,值得单独记一笔:** 它意味着一个 residue 数多于 OpenFHE 模板的 GPU 密文,**多出来的部分被静默丢掉**,然后 `:102` 的 `SetLevel` 再按截断后的数目算回去。没有报错,没有警告,release 下连那个 assert 都没有。

这正是 `CoeffsToSlots.cu:120` 那句话的形状——*the value survives, but it comes out wrong*。之前实测过 bootstrap 产出的密文带 `SPECIALlimb = 9` 而 fresh 密文是 `0`,所以"GPU 密文 residue 比模板多"不是假想。

我没有证据说这条路径正在 bootstrap 里被走到(测试用的模板 `c2` 是 fresh 的,towers 满的),所以**不列为 27 bit 的候选**。但一个静默截断加一个 release 下失效的 assert,本身就该修——至少把 assert 换成 throw。

## 3. 请求不变,加一条

1. 上面第 1 节那一行日志(零成本)
2. `OpenFHEBootstrapTest.ModRaise`(`:2308`)和 `OpenFHEBootstrap`(`:3395`),thormod / sparse 两组
3. `[FIDESlib] bootstrap precomputation:` 开头的全部行
4. CPU@mod59/60,**depth 23**(深度无关,别再用 37 把服务器跑崩)
