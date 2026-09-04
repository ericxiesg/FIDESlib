#ifndef API_LIGHTPLAINTEXT_HPP
#define API_LIGHTPLAINTEXT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "Definitions.hpp"

namespace fideslib {

/**
 * @brief A CKKS plaintext kept in its compact ("light") form.
 *
 * A normal encoded plaintext is @f$(L+1)@f$ RNS towers of @f$N@f$ 64-bit words - 16 MiB for
 * @f$N = 2^{16}, L = 30@f$. Encoding is however a level-independent step followed by a per-level
 * projection: the message is mapped to the @f$N@f$ integer coefficients of
 * @f$\lfloor \Delta \cdot \mathrm{IFFT}(m) \rceil@f$, and the towers are just that vector reduced
 * modulo each @f$q_i@f$ and NTT-transformed. Under FIXEDMANUAL scaling @f$\Delta@f$ does not depend
 * on the level, so the coefficient vector is the whole plaintext: @f$N@f$ int64 words, 0.5 MiB, and
 * it can be expanded at whatever level the ciphertext happens to sit at.
 *
 * That is what THOR's weights need. BERT-base has ~220k encoded weight plaintexts; at 16 MiB each
 * they are ~110 GiB and do not fit on a 32 GiB V100, at 0.5 MiB each they are ~110 GiB on disk but
 * only the handful in flight are ever expanded. This mirrors desilofhe's
 * `encode_to_light_plaintext` / `write_light_plaintext` / `read_light_plaintext`.
 *
 * Coefficients are *centred* (in @f$(-q_0/2, q_0/2)@f$) and stored in the natural coefficient order,
 * i.e. the order an OpenFHE DCRTPoly uses in `Format::COEFFICIENT`.
 */
class LightPlaintextImpl {
  public:
	/// @brief Assigns a fresh @ref uid; see that member for why it must never be left at 0.
	LightPlaintextImpl();

	/// @brief The N centred integer coefficients of round(scale * IFFT(message)).
	std::vector<int64_t> coeffs;
	/// @brief Scaling factor the coefficients carry.
	double scale = 0.0;
	/// @brief Number of CKKS slots the message occupied.
	uint32_t slots = 0;
	/// @brief Noise scale degree (1 for a freshly encoded plaintext).
	uint32_t noise_scale_deg = 1;
	/**
	 * @brief The level this plaintext is meant to be used at, or -1 when it is level-agnostic.
	 * Informational: expansion always follows the ciphertext. THOR encodes weights at a fixed level
	 * per stage, and keeping the hint lets a mismatch be reported instead of silently costing scale.
	 */
	int32_t level_hint = -1;
	/**
	 * @brief Identity used to key the expansion cache; unique per process, assigned by the constructor.
	 *
	 * Every construction path has to set this. A light plaintext that shares a uid with another one is
	 * served that other one's expansion out of `CryptoContextImpl::light_plaintext_cache` - a silently
	 * wrong weight, not a crash. Deserialisation used to leave it at 0, which made every weight read
	 * back from disk alias the first one.
	 */
	uint64_t uid;

	/// @brief Bytes the compact form occupies (what the caller saves by not expanding).
	[[nodiscard]] size_t Bytes() const { return coeffs.size() * sizeof(int64_t); }

	/// @brief Serialise to `path` (little-endian, see LightPlaintext.cpp for the layout).
	void Save(const std::string& path) const;
	/// @brief Read back a file written by Save(). Throws on a bad magic/version or a short read.
	static LightPlaintext Load(const std::string& path);
};

} // namespace fideslib

#endif // API_LIGHTPLAINTEXT_HPP
