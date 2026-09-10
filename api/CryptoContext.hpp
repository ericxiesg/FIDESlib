#ifndef API_CRYPTOCONTEXT_HPP
#define API_CRYPTOCONTEXT_HPP

#include <any>
#include <map>
#include <complex>
#include <cstdint>
#include <functional>
#include <array>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "CCParams.hpp"
#include "Ciphertext.hpp"
#include "Definitions.hpp"
#include "LightPlaintext.hpp"
#include "KeyPair.hpp"
#include "Plaintext.hpp"
#include "PublicKey.hpp"
#include "Serialize.hpp"

namespace fideslib {

/// @brief Specialization of CryptoContext for the DCRTPoly representation.
template <> class CryptoContextImpl<DCRTPoly> {

  public:
	CryptoContextImpl() = default;
	~CryptoContextImpl();

	// ---- Copy ----

	CryptoContextImpl(const CryptoContextImpl&)			   = delete;
	CryptoContextImpl& operator=(const CryptoContextImpl&) = delete;

	// ---- Move ----

	CryptoContextImpl(CryptoContextImpl&&)			  = default;
	CryptoContextImpl& operator=(CryptoContextImpl&&) = delete;

	// ---- Context Setup ----

	/// @brief Enable a particular feature in the context.
	void Enable(PKESchemeFeature feature);
	void Enable(uint32_t featureMask);

	// ---- Getters ----

	uint32_t GetCyclotomicOrder() const;
	uint32_t GetRingDimension() const;
	double GetPreScaleFactor(uint32_t slots);

	// ---- Setters ----
	void SetAutoLoadPlaintexts(bool autoload);
	void SetAutoLoadCiphertexts(bool autoload);
	void SetDevices(const std::vector<int>& devices);

	// ---- Load to devices ----

	/// @brief Load the context to the devices.
	void LoadContext(const PublicKey<DCRTPoly>& publicKey);
	/// @brief Load a plaintext to the devices.
	/// @param pt Plaintext to load.
	void LoadPlaintext(Plaintext& pt);
	/// @brief Load a ciphertext to the devices.
	/// @param ct Ciphertext to load.
	void LoadCiphertext(Ciphertext<DCRTPoly>& ct);

	// ---- Key Generation ----

	/// @brief Generate a public/private key pair.
	KeyPair<DCRTPoly> KeyGen();
	/// @brief Generate the evaluation multiplication keys.
	void EvalMultKeyGen(const PrivateKey<DCRTPoly>& sk);
	/// @brief Generate the evaluation rotation keys for the given steps.
	void EvalRotateKeyGen(const PrivateKey<DCRTPoly>& sk, const std::vector<int32_t>& steps);

	// ---- Bootstrapping ----

	/// @brief Generate bootstrap precomputation data.
	void EvalBootstrapSetup(const std::vector<uint32_t>& levelBudget = { 5, 4 },
	  std::vector<uint32_t> dim1									 = { 0, 0 },
	  uint32_t slots												 = 0,
	  uint32_t correctionFactor										 = 0,
	  bool precompute												 = true,
	  bool btsfirstboot												 = false);
	/// @brief Generate the evaluation bootstrap keys.
	void EvalBootstrapKeyGen(const PrivateKey<DCRTPoly>& secretKey, uint32_t slots);

	// ---- Serialization ----
	static bool SerializeEvalMultKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");
	static bool SerializeEvalAutomorphismKey(std::ostream& ser, const SerType& sertype, const std::string& keyTag = "");

	// ---- Deserialization ----
	bool DeserializeEvalMultKey(std::istream& ser, const SerType& sertype) const;
	bool DeserializeEvalAutomorphismKey(std::istream& ser, const SerType& sertype) const;

	// ---- Encoding ----

	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<std::complex<double>>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);
	Plaintext
	MakeCKKSPackedPlaintext(const std::vector<double>& value, size_t noiseScaleDeg = 1, uint32_t level = 0, std::shared_ptr<void> params = nullptr, uint32_t slots = 0);

	// ---- Encryption ----

	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PublicKey<DCRTPoly>& pk);
	Ciphertext<DCRTPoly> Encrypt(const PublicKey<DCRTPoly>& pk, Plaintext& pt);
	Ciphertext<DCRTPoly> Encrypt(Plaintext& pt, const PrivateKey<DCRTPoly>& sk);
	Ciphertext<DCRTPoly> Encrypt(const PrivateKey<DCRTPoly>& sk, Plaintext& pt);
	DecryptResult Decrypt(Ciphertext<DCRTPoly>& ct, const PrivateKey<DCRTPoly>& sk, Plaintext* pt);
	DecryptResult Decrypt(const PrivateKey<DCRTPoly>& sk, Ciphertext<DCRTPoly>& ct, Plaintext* pt);

	// ---- Operations ----

	Ciphertext<DCRTPoly> EvalNegate(const Ciphertext<DCRTPoly>& ct);
	void EvalNegateInPlace(Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAdd(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalAdd(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalAddInPlace(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalAddInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalAddInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalAddMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalAddMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalAddMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalAddMany(const std::vector<Ciphertext<DCRTPoly>>& ciphertexts);
	void EvalAddManyInPlace(std::vector<Ciphertext<DCRTPoly>>& ciphertexts);

	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSub(Plaintext& pt, const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSub(const Ciphertext<DCRTPoly>& ct, double scalar);
	Ciphertext<DCRTPoly> EvalSub(double scalar, const Ciphertext<DCRTPoly>& ct);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	void EvalSubInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalSubInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalSubMutable(Ciphertext<DCRTPoly>& ct, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalSubMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct);
	void EvalSubMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMult(Plaintext& pt, const Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct1, double scalar);
	Ciphertext<DCRTPoly> EvalMult(double scalar, const Ciphertext<DCRTPoly>& ct1);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	void EvalMultInPlace(Ciphertext<DCRTPoly>& ct1, double scalar);
	void EvalMultInPlace(double scalar, Ciphertext<DCRTPoly>& ct1);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);
	Ciphertext<DCRTPoly> EvalMultMutable(Ciphertext<DCRTPoly>& ct1, Plaintext& pt);
	Ciphertext<DCRTPoly> EvalMultMutable(Plaintext& pt, Ciphertext<DCRTPoly>& ct1);
	void EvalMultMutableInPlace(Ciphertext<DCRTPoly>& ct1, Ciphertext<DCRTPoly>& ct2);

	Ciphertext<DCRTPoly> EvalSquare(const Ciphertext<DCRTPoly>& ct);
	void EvalSquareInPlace(Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalSquareMutable(Ciphertext<DCRTPoly>& ct);

	// ---- THOR-style primitives (see docs/thor-engine-requirements.md) ----
	/// @brief Generate the conjugation key (automorphism index 2N-1). EvalBootstrapKeyGen also generates it.
	void EvalConjugateKeyGen(const PrivateKey<DCRTPoly>& sk);
	/// @brief Complex conjugation of every slot (one key switch, no level consumed).
	Ciphertext<DCRTPoly> EvalConjugate(const Ciphertext<DCRTPoly>& ct);
	/// @brief Multiply every slot by the imaginary unit i (monomial X^{N/2}); free: no key switch, no level, no scale change.
	Ciphertext<DCRTPoly> EvalMultByI(const Ciphertext<DCRTPoly>& ct);
	/// @brief Multiply by a small integer without consuming a level or changing the scale (THOR's DeltaCiphertext trick).
	Ciphertext<DCRTPoly> EvalMultByInteger(const Ciphertext<DCRTPoly>& ct, uint64_t k);
	/// @brief Drop `levels` RNS limbs without rescaling (THOR's level_down).
	Ciphertext<DCRTPoly> EvalLevelReduce(const Ciphertext<DCRTPoly>& ct, uint32_t levels);
	/// @brief Remaining multiplicative levels of a ciphertext (0 = only q0 left).
	uint32_t GetRemainingLevels(const Ciphertext<DCRTPoly>& ct) const;

	// ---- Light plaintexts (docs/light_plaintext.md) ----
	/**
	 * @brief Encode a message into its compact, level-agnostic form: the N centred integer coefficients
	 * of round(Delta * IFFT(message)) instead of the (L+1) RNS towers. 0.5 MiB instead of 16 MiB at
	 * N = 2^16, L = 30. THOR's `encode_to_light_plaintext`.
	 *
	 * Requires a scaling technique whose Delta does not depend on the level (FIXEDMANUAL / FIXEDAUTO);
	 * under FLEXIBLE* the coefficients are only valid at `level_hint` and expanding elsewhere throws.
	 * @param levelHint Level the plaintext is intended for, recorded for that check. -1: any level.
	 */
	LightPlaintext MakeLightPlaintext(const std::vector<std::complex<double>>& value, uint32_t slots = 0, int32_t levelHint = -1);
	/// @brief Real-valued overload of MakeLightPlaintext.
	LightPlaintext MakeLightPlaintext(const std::vector<double>& value, uint32_t slots = 0, int32_t levelHint = -1);
	/**
	 * @brief Expand a light plaintext into a usable plaintext at `level` (consumed levels, OpenFHE
	 * convention). On the GPU backend the towers are built on the device from the coefficient vector,
	 * so only N * 8 bytes cross PCIe.
	 */
	Plaintext ExpandLightPlaintext(const LightPlaintext& lp, uint32_t level);
	/// @brief Multiply by a light plaintext, expanding it at the ciphertext's level (cached).
	Ciphertext<DCRTPoly> EvalMult(const Ciphertext<DCRTPoly>& ct, const LightPlaintext& lp);
	/// @brief Add a light plaintext, expanding it at the ciphertext's level (cached).
	Ciphertext<DCRTPoly> EvalAdd(const Ciphertext<DCRTPoly>& ct, const LightPlaintext& lp);
	/// @brief Drop every cached expansion (call between stages to bound device memory).
	void ClearLightPlaintextCache();

	/**
	 * @brief Drop pooled auxiliary polynomials, keeping at most @p keep of them.
	 *
	 * A destroyed ciphertext does not free its polynomials; it parks them in the context's auxiliary
	 * pool so the next one can reuse them. Nothing ever drains that pool, so it settles at the
	 * high-water mark of simultaneously live ciphertexts and holds it for the rest of the run.
	 *
	 * Trimming it is what lets memory actually leave: the discarded polynomials free their limbs to
	 * the device pool's per-size-class free lists, and a slab all of whose blocks are free can then go
	 * back to the driver. Without this step reclamation finds nothing to return. Call it at a stage
	 * boundary, where the working set is genuinely smaller than it was mid-stage.
	 */
	void TrimAuxiliaryPolys(size_t keep = 0);

	/// @brief How many auxiliary polynomials the pool is currently holding.
	size_t GetAuxiliaryPolyCount() const;
	/**
	 * @brief Device pool bytes: {pooled, in_use, driver_free, driver_total}. All zero on a CPU context.
	 *
	 * Every memory fix so far was judged by whether a run got further, which is one bit per run. These
	 * four separate what that bit conflates: a large `pooled - in_use` means the pool is hoarding and
	 * the circuit is not the problem, while a small `driver_free` with a small `pooled` means the keys
	 * and plaintexts are. Cheap enough to call at every stage boundary.
	 */
	std::array<size_t, 4> GetDeviceMemory() const;
	/// @brief Number of expansions currently cached.
	size_t GetLightPlaintextCacheSize() const;
	/// @brief ExpandLightPlaintext through the FIFO cache.
	Plaintext GetExpandedLightPlaintext(const LightPlaintext& lp, uint32_t level);
	/// @brief Levels a ciphertext has already consumed, i.e. the `level` MakeCKKSPackedPlaintext expects.
	uint32_t GetConsumedLevels(const Ciphertext<DCRTPoly>& ct) const;
	/**
	 * @brief Level plan for the rotation keys registered with EvalRotateKeyGen: index -> maximum number of
	 * remaining levels the key will ever be applied at (THOR's `create_fixed_rotation_key(sk, delta, level)`).
	 * Keys are stored level-truncated on the device; keys absent from the map are stored complete.
	 * Must be called before LoadContext. key_level_margin is added on top.
	 */
	void SetRotationKeyLevels(const std::map<int32_t, uint32_t>& maxRemainingLevels);

	/// @brief Multiplication without relinearisation (degree-2 result). Add/sub/pt-mult/rescale/level-reduce work on it; relinearise once.
	Ciphertext<DCRTPoly> EvalMultNoRelin(const Ciphertext<DCRTPoly>& ct1, const Ciphertext<DCRTPoly>& ct2);
	/// @brief Squaring without relinearisation.
	Ciphertext<DCRTPoly> EvalSquareNoRelin(const Ciphertext<DCRTPoly>& ct);
	/// @brief Key-switch the degree-2 component away (one key switch). Returns the input unchanged for degree-1 ciphertexts.
	Ciphertext<DCRTPoly> EvalRelinearize(const Ciphertext<DCRTPoly>& ct);

	Ciphertext<DCRTPoly> EvalRotate(const Ciphertext<DCRTPoly>& ciphertext, int32_t index);
	void EvalRotateInPlace(Ciphertext<DCRTPoly>& ciphertext, int32_t index);

	std::shared_ptr<void> EvalFastRotationPrecompute(const Ciphertext<DCRTPoly>& ct);
	Ciphertext<DCRTPoly> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, int32_t index, uint32_t m, const std::shared_ptr<void>& precomp);
	Ciphertext<DCRTPoly> EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, int32_t index, const std::shared_ptr<void>& digits, bool addFirst);
	std::vector<Ciphertext<DCRTPoly>> EvalFastRotation(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, uint32_t m, const std::shared_ptr<void>& precomp);
	std::vector<Ciphertext<DCRTPoly>>
	EvalFastRotationExt(const Ciphertext<DCRTPoly>& ct, const std::vector<int32_t>& indices, const std::shared_ptr<void>& digits, bool addFirst);

	Ciphertext<DCRTPoly> EvalChebyshevSeries(const Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	void EvalChebyshevSeriesInPlace(Ciphertext<DCRTPoly>& ct, std::vector<double>& coeffs, double a, double b);
	static std::vector<double> GetChebyshevCoefficients(std::function<double(double)>& func, double a, double b, size_t degree);

	Ciphertext<DCRTPoly> Rescale(const Ciphertext<DCRTPoly>& ciphertext);
	void RescaleInPlace(Ciphertext<DCRTPoly>& ciphertext);

	static void SetLevel(Ciphertext<DCRTPoly>& ct, size_t level);

	Ciphertext<DCRTPoly> EvalBootstrap(const Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);
	void EvalBootstrapInPlace(Ciphertext<DCRTPoly>& ciphertext, uint32_t numIterations = 1, uint32_t precision = 0, bool prescaled = false);

	Ciphertext<DCRTPoly> AccumulateSum(const Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride = 1);
	void AccumulateSumInPlace(Ciphertext<DCRTPoly>& ct, int slots, int stride, int start);

	void ConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct, int gStep, int bStep, const std::vector<Plaintext>& pts, const std::vector<int>& indexes, int stride = 1, int rowSize = 0);

	void SpecialConvolutionTransformInPlace(Ciphertext<DCRTPoly>& ct,
	  int gStep,
	  int bStep,
	  const std::vector<Plaintext>& pts,
	  Plaintext& mask,
	  const std::vector<int>& indexes,
	  int stride			 = 1,
	  int maskRotationStride = 1,
	  int rowSize			 = 0);

  public:
	// ---- Internal State ----

	std::any cpu;
	std::any gpu;
	/// @brief Whether the context has been loaded to the devices.
	bool loaded = false;
	/// @brief List of devices the context is loaded on.
	std::vector<int> devices = { 0 };
	/// @brief Store bootstrap rotation keys level-truncated on the device (memory saving, see docs/level_truncated_keys.md).
	bool truncate_keys = true;
	/// @brief Safety margin (levels) added on top of the computed usage level of truncated keys.
	int key_level_margin = 1;
	/// @brief Reload a truncated key when it is used above its plan level instead of throwing (env FIDESLIB_KEY_GROW=1).
	bool allow_key_grow = false;
	/// @brief Device bytes currently held by key-switching keys on the loaded context.
	size_t GetKeyDeviceBytes() const;
	/// @brief Number of truncated keys that had to be grown at runtime (0 == level plan was sufficient).
	int GetGrownKeyCount() const;
	/// @brief Whether plaintexts should be automatically loaded to the device upon encryption.
	bool auto_load_plaintexts = false;
	/// @brief Whether ciphertexts should be automatically loaded to the device upon creation.
	bool auto_load_ciphertexts = true;
	/// @brief Self reference to enable shared_from_this-like behavior.
	std::weak_ptr<CryptoContextImpl<DCRTPoly>> self_reference;
	/// @brief Multiplicative depth of the context.
	uint32_t multiplicative_depth = 0;
	/// @brief Rotation indexes for which rotation keys are available.
	std::vector<int32_t> rotation_indexes;
	/// @brief Rotation index -> max remaining levels (see SetRotationKeyLevels).
	std::map<int32_t, uint32_t> rotation_key_levels;
	/// @brief How many expanded light plaintexts to keep around (FIFO). 0 disables caching.
	size_t light_plaintext_cache_capacity = 64;
	/// @brief (light plaintext uid, level) -> expanded plaintext.
	std::map<std::pair<uint64_t, uint32_t>, Plaintext> light_plaintext_cache;
	/// @brief Insertion order of light_plaintext_cache, for FIFO eviction.
	std::vector<std::pair<uint64_t, uint32_t>> light_plaintext_cache_order;
	/// @brief Bootstrap slots available.
	std::vector<uint32_t> slots_bootstrap;
	/// @brief Secret key distribution.
	SecretKeyDist keyDist = UNIFORM_TERNARY;

	// ---- Copy helpers ----

	uint32_t CopyDeviceCiphertext(const CiphertextImpl<DCRTPoly>& ct);

	// --- Map Handling ----

	/// @brief  Registry of plaintexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_plaintexts;
	/// @brief  Registry of ciphertexts stored on the GPU (opaque types).
	std::unordered_map<uint32_t, std::shared_ptr<void>> device_ciphertexts;
	/// @brief Next available handle for GPU objects. Zero is reserved as a null handle.
	uint32_t next_gpu_handle = 1;

	uint32_t RegisterDevicePlaintext(std::shared_ptr<void>&& p);
	uint32_t RegisterDeviceCiphertext(std::shared_ptr<void>&& c);
	std::shared_ptr<void>& GetDevicePlaintext(uint32_t handle);
	std::shared_ptr<void>& GetDeviceCiphertext(uint32_t handle);
	bool EvictDevicePlaintext(uint32_t handle);
	bool EvictDeviceCiphertext(uint32_t handle);

	void Synchronize() const;

	static std::vector<int> GetConvolutionTransformRotationIndices(int rowSize, int bStep, int stride, uint32_t gStep);
};

} // namespace fideslib

#endif // API_CRYPTOCONTEXT_HPP