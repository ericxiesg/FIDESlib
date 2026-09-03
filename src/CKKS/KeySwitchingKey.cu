//
// Created by carlosad on 26/09/24.
//

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <source_location>
#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
// constexpr int PREFIX_SIZE = 0;
#else
#include <source_location>
using sc = std::source_location;
// constexpr int PREFIX_SIZE = 23;
#endif

namespace FIDESlib::CKKS {

void KeySwitchingKey::Initialize(RawKeySwitchKey& rkk) {
	Initialize(rkk, -1, nullptr);
}

void KeySwitchingKey::Initialize(RawKeySwitchKey& rkk, int maxLevel_, Reloader reloader_) {
	CudaNvtxRange r(std::string{ sc::current().function_name() }.substr());
	CKKS::SetCurrentContext(cc);
	keyID = rkk.keyid;

	// Multi-GPU keys use a different layout (grown plain limbs); keep them complete.
	if (cc->GPUid.size() > 1 || maxLevel_ >= cc->L)
		maxLevel_ = -1;
	maxLevel = maxLevel_;
	reloader = std::move(reloader_);

	a.generateDecompAndDigit(true, maxLevel);
	b.generateDecompAndDigit(true, maxLevel);
	if (cc->GPUid.size() > 1) {
		a.grow(cc->L, false, true);
		b.grow(cc->L, false, true);
	}
	// loadDecompDigit only fills the limbs that exist, so a truncated key is loaded correctly.
	a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
	b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);

	cudaDeviceSynchronize();
}

/** Human-readable identification of a key, for the ensureLevel diagnostics. */
std::string KeySwitchingKey::describe() const {
	return "key '" + std::string(keyID) + "'" + (index != INT32_MIN ? " (rotation index " + std::to_string(index) + ")" : std::string{});
}

void KeySwitchingKey::ensureLevel(int level) {
	if (maxLevel < 0 || level <= maxLevel)
		return;

	// A truncated key used above its plan level is almost always a mistake in the (delta -> level) table the
	// caller passed to SetRotationKeyLevels, and reloading it silently turns that mistake into a per-call
	// host->device key transfer. Report it precisely instead, unless growth was explicitly asked for.
	if (!cc->allowKeyGrow || !reloader) {
		throw std::runtime_error("KeySwitchingKey::ensureLevel: " + describe() + " was loaded truncated to level " + std::to_string(maxLevel) +
								 " but is applied to a ciphertext at level " + std::to_string(level) + ". Fix the level plan (SetRotationKeyLevels / " +
								 "GetBootstrapKeyLevelPlan), raise ContextData::keyLevelMargin" +
								 (reloader ? ", or set FIDESLIB_KEY_GROW=1 to reload the key on demand." : " (this key has no reloader, so it cannot grow)."));
	}

	CKKS::SetCurrentContext(cc);
	std::cerr << "[FIDESlib] warning: reloading level-truncated " << describe() << " for level " << level << " (was truncated to " << maxLevel
			  << "); fix the level plan to avoid the reload cost" << std::endl;

	rebuildAtLevel(std::min(level, cc->L));
	grown = true;
}

void KeySwitchingKey::rebuildAtLevel(int newMaxLevel) {
	if (!reloader)
		throw std::runtime_error("KeySwitchingKey::rebuildAtLevel: " + describe() + " has no reloader");

	// The key limbs about to be freed may still be referenced by pointer tables of queued kernels.
	cudaDeviceSynchronize();
	CudaCheckErrorMod;

	RawKeySwitchKey rkk = reloader();

	// Rebuild through the exact path Initialize() uses, so the key ends up with the same limb layout a
	// key loaded at this level from the start would have. See LimbPartition::resetDecompAndDigit.
	const int wanted = (newMaxLevel >= cc->L) ? -1 : newMaxLevel;
	a.resetDecompAndDigit();
	b.resetDecompAndDigit();
	a.generateDecompAndDigit(true, wanted);
	b.generateDecompAndDigit(true, wanted);
	a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
	b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);

	cudaDeviceSynchronize();
	CudaCheckErrorMod;

	maxLevel = wanted;
	assert(coversLevel(newMaxLevel));
}

bool KeySwitchingKey::coversLevel(int level) const {
	return a.decompDigitLevelCovered() >= level && b.decompDigitLevelCovered() >= level;
}

size_t KeySwitchingKey::deviceBytes() const {
	return a.decompDigitDeviceBytes() + b.decompDigitDeviceBytes();
}

KeySwitchingKey::KeySwitchingKey(Context& cc)
: my_range(loc, LIFETIME), keyID(""), cc((assert(cc != nullptr), CudaNvtxStart(std::string{ sc::current().function_name() }.substr()), cc)),
  a(*cc, -1, false, true), b(*cc, -1, false, true) {
	CudaNvtxStop();
}
} // namespace FIDESlib::CKKS
