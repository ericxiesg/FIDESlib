# 回复：你是对的，我修漏了——四处全修了，但有一处你的建议会改坏

日期：2026-09-11。**C++ 未编译。**

## 一、这是我的疏漏

`43e1c56` 我只修了 `EvalNegate`，**没有去 grep `multScalar(-1.0)` 的其它调用点**。
定位到机制之后不把同机制的调用点查全，是个方法上的错误，不是运气问题。你查全了，谢谢。

而且你那条因果链完全正确，比我的推断更准：

* `he_inv` 走的是 **float 分支** `subtract(float, ct)` → `EvalScalarSub` → `EvalSub(double, ct)`；
* `113e854` 改的是 **numpy 分支**——所以 v6 ≡ v8，这条对照是决定性的；
* `he_inv` 在 softmax 里，`he_invsqrt` 在 LayerNorm 里，所以**第一个坏掉的是 softmax 而不是 norm_1**。

我上一份说「bootstrap 是通往 `07a` 路径上唯一没验证过的算子」——那句话本身没错，但它只对
「如果 `07a` 就已经坏了」成立。`he_inv` 在 `07c`/`07d`，在 `07a` 之后。**探针本来会打出
`07a` 正常、`07d` 崩掉**，我不该在拿到探针数据前就把 bootstrap 说成首要嫌疑。

## 二、四处都修了，但第 1 处按你的建议改会引入 sign bug

我用**每个函数的 CPU 回退分支**来定语义——它委托给 OpenFHE，是权威。

| 行 | 函数 | CPU 分支 | 原 GPU 路径 | 结论 |
|---|---|---|---|---|
| 1073 | `EvalSub(Plaintext& pt, ct)` | `EvalSub(ptImpl, ctImpl)` = `pt - ct` | `-ct + pt` = `pt - ct` | **sign 本来就是对的** |
| 1122 | `EvalSub(double scalar, ct)` | `EvalSub(scalar, ctImpl)` = `scalar - ct` | `-ct + scalar` = `scalar - ct` | sign 对（我之前已删掉尾部那次取负） |
| 1188/1190 | `EvalSubInPlace(double scalar, ct1)` | `EvalSubInPlace(scalar, ct1Impl)` | `-ct + s` 再取负 = `ct - s` | **sign 是错的** |

**第 1 处：`subPt` 会把它改坏。** 那个函数的签名是 `EvalSub(Plaintext& pt, const Ciphertext& ct)`，
语义是 `pt - ct`；而 `subPt` 算的是 `ct - pt`。**换成 `subPt` 会把符号翻过来。**
这里只需要把 `multScalar(-1.0)` 换成 `negate()`，addPt 保留。

**第 3 处的 sign 判断我同意你的怀疑，而且能定死**：紧邻的另一个重载
`EvalSubInPlace(ct, scalar)` 已经是 `addScalar(-scalar)` = `ct - scalar`。
两个参数顺序相反的重载算出同一个东西，那一定有一个是错的——而且这正是我之前在
**非 in-place** 版本上修掉的同一个 bug（同样的 negate/add/negate 三段式，同样算成 `ct - scalar`）。
所以它应当是 `scalar - ct`，即 `negate(); addScalar(scalar);`——**sign 和 scale 一起修**。

你给的 `addScalar(-scalar)` 只修 scale、保留 `ct - scalar`，那会把这个重载永久固定成
另一个重载的复制品。我选了改 sign。**如果你手上有 OpenFHE 源码，请核一下
`EvalSubInPlace(double, Ciphertext&)` 的语义再定**——这是这次唯一一处我改了行为的地方。

## 三、改完之后

`api/` 里 `multScalar(-1.0)` **一处不剩**（`grep` 只剩注释里提到它的两行）。
`src/` 里本来就没有。

`negate()` 是 scale-中性的，而 `addScalar` 本来就会读密文当前的 `NoiseLevel` 来编码常数，
所以这两个接起来在任何 scale 下都对。

## 四、为什么 `4a2023d` 的守卫没抓到

你的解释是对的：守卫只看 `addPt`/`subPt`，而这条路径走的是 `multScalar` + `addScalar`，
`addScalar` 又是**自适应**的（读 `NoiseLevel`），所以它安静地用了一个错的 `NoiseLevel`，
不会抛。守卫没白加——它排除了明文加法这一整类——但它覆盖不到标量路径。

## 五、下一跑

**先重编。** 探针仍然建议开着：这次如果 softmax 好了，`07a`–`07d` 四行就是新的基准剖面；
如果还没好，它们直接指出是哪一段。

```bash
python3 -u -m thorfhe.bench fhe --engine fideslib --device cuda:0 \
    --depth 37 --dnum 4 --bootstrap-level-budget 3,3 \
    --binary-rotations --refresh-after-dense \
    --layers 1 --limit 1 --per-stage --device-memory --offline
```

---

## 本次改动

| 文件 | 改动 |
|---|---|
| `api/CryptoContext.cpp` | `EvalSub(pt, ct)`、`EvalSub(double, ct)`、`EvalSubInPlace(double, ct)` 全部改用 `negate()`；最后一个同时修 sign **未编译** |
| `../patch/` | PR2 仍 10 个文件；重建校验与前向引用校验都重跑通过 |
