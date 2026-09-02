//
// Created by carlosad on 26/09/24.
//

#ifndef GPUCKKS_KEYSWITCHINGKEY_CUH
#define GPUCKKS_KEYSWITCHINGKEY_CUH

#include <cinttypes>
#include <functional>
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
	Context& cc;
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

	explicit KeySwitchingKey(Context& cc);

	void Initialize(RawKeySwitchKey& rkk);
	/** Initialize keeping only limbs needed for ciphertext levels <= maxLevel. maxLevel < 0 -> full key. */
	void Initialize(RawKeySwitchKey& rkk, int maxLevel, Reloader reloader = nullptr);

	/**
	 * Guarantee the key can be applied to a ciphertext at `level`. If the key was truncated below that
	 * level it is grown in place (requires `reloader`; throws otherwise). Cheap when already sufficient.
	 */
	void ensureLevel(int level);

	[[nodiscard]] bool isTruncated() const { return maxLevel >= 0; }
	/** Device memory currently held by this key (a + b). */
	[[nodiscard]] size_t deviceBytes() const;
};

} // namespace CKKS
} // namespace FIDESlib

#endif // GPUCKKS_KEYSWITCHINGKEY_CUH
