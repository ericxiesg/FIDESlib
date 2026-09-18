# LayerNorm 的方差窗口做成每层标定：省 4 个 level，并修掉最后一层

日期：2026-09-18。承接 `2e78bcd`。

---

## 0. 结论

`VARIANT_BOUNDS` 是**一个窗口套十二层**，而且是照第 0 层配的。我在**明文模型**上量了每层
LayerNorm 真正拿到的逐 token 方差（LN1 是 `hidden + dense`，LN2 是 `norm_1 + output`），
200 条 MRPC validation，不用 FHE、不花 level：

```
layer  LN1 var [min, max]      ratio    LN2 var [min, max]      ratio
0      [0.2228,  0.6868]         3.1    [1.002,    7.566]         7.6
1      [0.2032,  0.8268]         4.1    [0.8749,   8.749]        10.0
2      [0.2184,  1.004 ]         4.6    [0.8972, 111.5  ]       124.3
...
9      [0.09679, 1.244 ]        12.9    [0.6591, 1832   ]      2780.1
10     [0.039,   1.264 ]        32.4    [0.6816, 1638   ]      2403.2
11     [0.02297, 1.361 ]        59.3    [0.6971,   3.987]         5.7
```

**实测收益（真实 checkpoint，`--layers 1`）：**

| | 之前 | 现在 |
|---|---|---|
| layer 0 出来时的 level | **1** | **5** |
| hidden relRMSE | 2.776e-03 | 2.784e-03 |
| label agreement | 100% | 100% |

**省 4 个 level，精度不变。** 一层吃 36 个、只有 37 个可用，所以 4 个不是零头。
顺带：layer 1 现在**不靠边界 refresh 也能跑**（虽然 refresh 仍然是稳妥的做法，默认保留）。

---

## 1. 三个由这张表得到的结论

### 1.1 旧窗口在最后一层是**错的**

LN1 下界 0.15，而 layer 11 量到 0.023——乘回 `ACTIVATION_SCALE² = 4` 是 **0.092，在下界之下**。
低于下界 `he_invsqrt` 不收敛。layer 10 是 0.156，**只比下界高 4%**——在一个自举带 0.017
绝对误差的设备上，这不叫余量。

### 1.2 旧窗口在前面几层**太宽**，宽出来的就是 level

`he_invsqrt` 的迭代次数来自窗口的**比值**，每次迭代两个 level。layer 0 实际比值 3.1，
而旧窗口给的是 67。这就是那 4 个 level 的来源。

### 1.3 它解释了 variant 3 的存在，并且取代了它

`he_layernorm2` 和 `he_layernorm3` **除了默认 bounds 完全一样**（`HALVES[2] == HALVES[3]`）。
THOR 把 layer 9、10 路由到更宽的那个——而这两层正是 LN2 方差冲到 **1832 和 1638** 的地方，
其余各层是 3.99 到 111。所以"variant 3 给 9 和 10"就是这张表的**两格近似**。
现在按层查表，variant 的选择不再携带信息。

---

## 2. 安全边界

- 窗口 = 实测值，**两边各放宽 2 倍**（比值放宽 4 倍）。16 条句子时 layer 9 的 LN2 上界是 625，
  200 条时是 **1832**——样本会动，所以这个 margin 不是装饰。放宽 2 倍还能省 3.2 level，
  放宽 4 倍就一点不省了，2 是拐点。
- 表里没有的层（别的 geometry、别的 checkpoint）落回原来的 `VARIANT_BOUNDS`。
- 明文量的，所以换 checkpoint 重量一次就行，几秒钟，不用 GPU。
- **这是在 MRPC 上量的。** 换数据集要重量；量程检查会在 ClearEngine 上拦住，设备上不会。

---

## 3. ⚠️ 对你的影响：level 变了，**旋转密钥计划会变**

`plan_rotations` 是拿同一份代码干跑出来的，所以它会自动跟着变——但**你已经生成的旋转密钥
是按旧 level 截断的**。拉了这个改动之后：

> **请重新生成旋转密钥**（不要复用旧的 key cache）。

`--plaintext-cache` 不受影响（明文编码和 level 无关），可以继续用。

---

## 4. 还没解决的

12 层还是跑不完——修完 LayerNorm 窗口之后，更深的层仍然会撞 level。
per-layer 的 softmax 窗口（`Softmax.LAYERS`）是有的，但整条 level 预算在后面几层还不够。
这是下一个要量的东西，不过**不阻塞你现在的单层工作**。

另外实测：**`--refresh-after-dense` 还是必须的**，省下的 4 个 level 不足以去掉它。

221 passed, 67 skipped。

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
