#include "LightPlaintext.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace fideslib {

namespace {

/**
 * File layout (all little-endian, which is every platform this library builds on):
 *   char[8]  "FLPT0001"
 *   double   scale
 *   uint32   slots
 *   uint32   noise_scale_deg
 *   int32    level_hint
 *   uint32   n            (number of coefficients)
 *   int64[n] coeffs
 * 0.5 MiB + 32 bytes for N = 2^16. Deliberately raw: THOR reads hundreds of thousands of these.
 */
constexpr char MAGIC[8] = { 'F', 'L', 'P', 'T', '0', '0', '0', '1' };

template <typename T> void Put(std::ostream& os, const T& v) {
	os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T> void Get(std::istream& is, T& v, const std::string& path) {
	if (!is.read(reinterpret_cast<char*>(&v), sizeof(T)))
		throw std::runtime_error("LightPlaintext::Load: truncated file '" + path + "'");
}

} // namespace

void LightPlaintextImpl::Save(const std::string& path) const {
	std::ofstream os(path, std::ios::binary | std::ios::trunc);
	if (!os)
		throw std::runtime_error("LightPlaintext::Save: cannot open '" + path + "' for writing");

	os.write(MAGIC, sizeof(MAGIC));
	Put(os, scale);
	Put(os, slots);
	Put(os, noise_scale_deg);
	Put(os, level_hint);
	Put(os, static_cast<uint32_t>(coeffs.size()));
	if (!coeffs.empty())
		os.write(reinterpret_cast<const char*>(coeffs.data()), static_cast<std::streamsize>(coeffs.size() * sizeof(int64_t)));
	if (!os)
		throw std::runtime_error("LightPlaintext::Save: write failed for '" + path + "'");
}

LightPlaintext LightPlaintextImpl::Load(const std::string& path) {
	std::ifstream is(path, std::ios::binary);
	if (!is)
		throw std::runtime_error("LightPlaintext::Load: cannot open '" + path + "'");

	char magic[sizeof(MAGIC)];
	if (!is.read(magic, sizeof(magic)) || std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0)
		throw std::runtime_error("LightPlaintext::Load: '" + path + "' is not a light plaintext file");

	LightPlaintext lp = std::make_shared<LightPlaintextImpl>();
	Get(is, lp->scale, path);
	Get(is, lp->slots, path);
	Get(is, lp->noise_scale_deg, path);
	Get(is, lp->level_hint, path);
	uint32_t n = 0;
	Get(is, n, path);

	lp->coeffs.resize(n);
	if (n > 0 && !is.read(reinterpret_cast<char*>(lp->coeffs.data()), static_cast<std::streamsize>((size_t)n * sizeof(int64_t))))
		throw std::runtime_error("LightPlaintext::Load: truncated coefficients in '" + path + "'");

	return lp;
}

} // namespace fideslib
