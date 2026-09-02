汇报：Step 3 (lazy relin + Python wrapper) 调通结果
新 commit 00e3905 引入：degree-2 ciphertext (lazy relinearisation)、THOR API primitives、CMake arch auto-detect、pybind11 pyfideslib + 4 stage pytest。在服务器 (GV100/CUDA 12.9/OpenFHE 1.5.1.1) 上构建并测试。
已修复的 bug（共 5 个）
#	文件	bug	修复
1	api/CryptoContext.cpp	新增 THOR primitives 定义放在 } // namespace fideslib 之后，命名空间外编译失败	将 namespace 闭合移到文件末尾
2	src/CKKS/Ciphertext.cpp	relinearize() 用了不存在的 OPS::KEYSWITCH	在 OPS enum + opstr 加 RELINEARIZE 成员
3	api/CryptoContext.cpp	EvalMult(ct, pt) / EvalMultInPlace GPU 前的 CPU fallback 用 any_cast<ConstPlaintext>，但明文存的是 Plaintext → bad any_cast	改为 any_cast<Plaintext>
4	api/{Definitions.hpp,CCParams.*,bindings.cpp} + __init__.py	CKKS 默认 CKKSDataType=REAL，复数 slot 虚部丢失，complex roundtrip 失败	新增 SetCKKSDataType，Engine 默认 COMPLEX
5	api/CryptoContext.cpp	EvalSub(scalar, ct) GPU 路径多了一次 multScalar(-1.0)，结果符号反了 (ct-scalar 而非 scalar-ct)	删除多余的最后一次取负
-	python/tests/test_stage4_bootstrap.py	bootstrap 精度参数 50-bit scaling 太紧，err≈1.5e-2 > 1e-2	改为 59/60-bit（与 key-truncation 示例一致，err≈1e-5）
测试结果
- CPU pytest：16/16 全部通过（stage1 io / stage2 linear / stage3 lazy relin / stage4 bootstrap）
- CUDA pytest：stage1 (4/4) + stage2 前 4 项通过，随后 test_rotate segfault
⚠️ 严重 bug — 已停下，写好 bug note
CUDA 后端 level-truncated key 的按需 grow 会 segfault。当 SetRotationKeyLevels 把某 rotation key 截断加载后，若该 key 被用于高于其 maxLevel 的密文，KeySwitchingKey::ensureLevel → LimbPartition::growDecompAndDigitToLevel 的 grow 路径崩溃。完整 key (truncate_keys=False) 不受影响，CPU 后端不受影响。
复现：Engine("cuda:0", rotation_indexes={2:5}, ...); e.rotate(e.encrypt(x), 2) → SIGSEGV。
Bug note 已写到 D:\Workspace\THOR-v2\BUGNOTE_gpu_truncated_key_grow_segfault.md，含症状、复现、初步根因（grow 后 key limb 布局与 context 共享 aux poly/gather limb 不匹配，疑似越界 device 读）、影响和建议的下一步（compute-sanitizer 定位 + 审计 grow 与 generate 的对称性）。
需要你看一下这个 GPU grow 的 bug note 再决定怎么修。临时 workaround：CUDA 下设 FIDESLIB_KEY_TRUNCATION=0 用完整 key。