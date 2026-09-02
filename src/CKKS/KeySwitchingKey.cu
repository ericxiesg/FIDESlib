//
// Created by carlosad on 26/09/24.
//

#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/RNSPoly.cuh"
#include <algorithm>
#include <iostream>
#include <stdexcept>
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

void KeySwitchingKey::ensureLevel(int level) {
	if (maxLevel < 0 || level <= maxLevel)
		return;
	if (!reloader) {
		throw std::runtime_error("KeySwitchingKey::ensureLevel: key '" + std::string(keyID) + "' was truncated to level " + std::to_string(maxLevel) +
								 " but is needed at level " + std::to_string(level) + " and has no reloader");
	}
	CKKS::SetCurrentContext(cc);
	std::cerr << "[FIDESlib] warning: level-truncated key grown from level " << maxLevel << " to " << level
			  << " (increase ContextData::keyLevelMargin or fix the level plan to avoid the reload cost)" << std::endl;

	const int newLevel = std::min(level, cc->L);
	RawKeySwitchKey rkk = reloader();
	a.growDecompAndDigitToLevel(newLevel);
	b.growDecompAndDigitToLevel(newLevel);
	a.loadDecompDigit(rkk.r_key[0], rkk.r_key_moduli[0]);
	b.loadDecompDigit(rkk.r_key[1], rkk.r_key_moduli[1]);
	cudaDeviceSynchronize();

	maxLevel = (newLevel >= cc->L) ? -1 : newLevel;
	grown	 = true;
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
