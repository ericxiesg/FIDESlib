//
// Created by carlosad on 26/09/24.
//

#ifndef GPUCKKS_KEYSWITCHINGKEY_CUH
#define GPUCKKS_KEYSWITCHINGKEY_CUH

#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <memory>
#include <functional>
#include <string>
#include <vector>

#include "RNSPoly.cuh"
#include "openfhe-interface/RawCiphertext.cuh"

namespace FIDESlib {
namespace CKKS {

class KeySwitchingKey {
	static constexpr const char* loc{ "KeySwitchingKey" };
	CudaNvtxRange my_range;

  public:
	using Reloader = std::function<RawKeySwitchKey()>;

	KeyHash keyID;
	/**
	 * Non-owning, and deliberately neither a reference nor a shared_ptr.
	 *
	 * A `Context&` member dangles: CryptoContextImpl::LoadContext moves its local Context into a
	 * std::any, so the referent dies when LoadContext returns. Holding a `Context` by value fixes
	 * that but closes a reference cycle, because keys live inside ContextData::precom.keys: the
	 * ContextData can then never reach zero references, and every context dropped through the API
	 * keeps all of its device memory - eval and rotation keys, bootstrap plaintexts, auxiliary
	 * buffers - for the life of the process.
	 *
	 * A weak_ptr fixes the dangling reference without owning the object this key is a member of.
	 * It cannot expire while the key is alive, for exactly that reason, and context() asserts it.
	 */
	std::weak_ptr<ContextData> cc_weak;

	/** The context this key belongs to. Every use is a read of one of its fields. */
	[[nodiscard]] Context context() const {
		Context cc = cc_weak.lock();
		assert(cc != nullptr);
		return cc;
	}
	RNSPoly a;
	RNSPoly b;
	// std::vector<RNSPoly> mgpu_a;
	// std::vector<RNSPoly> mgpu_b;

	/**
	 * Level-truncated storage. maxLevel == -1 means the key holds every limb (legacy behaviour).
	 * Otherwise only the Q-limbs with prime id <= maxLevel (plus all special limbs) are resident on the
	 * device, which is sufficient to key-switch any ciphertext whose level is <= maxLevel.
	 */
	int maxLevel = -1;
	/** Optional callback that re-fetches the raw key (e.g. from the OpenFHE context) so a truncated key can grow. */
	Reloader reloader;
	/** Set once ensureLevel() had to grow this key. Lets callers/tests detect a too-tight level plan. */
	bool grown = false;
	/** Rotation index this key was built for, purely for diagnostics. INT32_MIN when not applicable. */
	int index = INT32_MIN;

	explicit KeySwitchingKey(Context& cc);

	void Initialize(RawKeySwitchKey& rkk);
	/** Initialize keeping only limbs needed for ciphertext levels <= maxLevel. maxLevel < 0 -> full key. */
	void Initialize(RawKeySwitchKey& rkk, int maxLevel, Reloader reloader = nullptr);

	/**
	 * Guarantee the key can be applied to a ciphertext at `level`. Cheap (a comparison) when the key already
	 * covers it. Otherwise the level plan is wrong: throws a diagnostic naming the key and both levels, or -
	 * when ContextData::allowKeyGrow is set and a reloader exists - reloads and rebuilds the key.
	 */
	void ensureLevel(int level);

	/** Reload the key from `reloader` and rebuild its limbs for ciphertext levels <= newMaxLevel (>= cc->L: complete). */
	void rebuildAtLevel(int newMaxLevel);

	/** True when every limb the key-switching kernels index at `level` is resident. */
	[[nodiscard]] bool coversLevel(int level) const;

	/** "key '<tag>' (rotation index n)", for diagnostics. */
	[[nodiscard]] std::string describe() const;

	[[nodiscard]] bool isTruncated() const { return maxLevel >= 0; }
	/** Device memory currently held by this key (a + b). */
	[[nodiscard]] size_t deviceBytes() const;
};

} // namespace CKKS
} // namespace FIDESlib

#endif // GPUCKKS_KEYSWITCHINGKEY_CUH
