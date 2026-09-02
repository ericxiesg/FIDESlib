// Level-truncated bootstrap keys: correctness + memory check.
//
// Runs the same CKKS bootstrap twice - once with level-truncated rotation keys (default) and once with
// complete keys - and compares (a) the bootstrapping error against the input, (b) the device memory held
// by the keys and (c) how many truncated keys had to be grown at runtime (must be 0 for a correct plan).
//
// Usage: key-truncation [logN=13] [slots=logN-1 bits] [depth=25] [dnum=3] [lb_e=3] [lb_d=3]
// Exit code 0 on success, 1 if the truncated run is less precise than the full run by more than a tolerance
// or if keys had to be grown.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <fideslib.hpp>

using namespace fideslib;

struct RunResult {
	double maxAbsError	 = 0.0;
	double bootstrapSec	 = 0.0;
	size_t keyBytes		 = 0;
	int grownKeys		 = 0;
	uint32_t levelsAfter = 0;
};

static RunResult RunBootstrap(bool truncateKeys, int logN, uint32_t numSlots, uint32_t depth, uint32_t dnum, std::vector<uint32_t> levelBudget, int repeat) {
	std::vector<int> devices = { 0 };

	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetSecretKeyDist(SPARSE_TERNARY);
	parameters.SetSecurityLevel(HEStd_NotSet);
	parameters.SetRingDim(1 << logN);
	parameters.SetNumLargeDigits(dnum);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetDevices(devices);
	parameters.SetScalingModSize(59);
	parameters.SetScalingTechnique(FLEXIBLEAUTO);
	parameters.SetFirstModSize(60);
	parameters.SetMultiplicativeDepth(depth);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	// Knobs under test (env vars FIDESLIB_KEY_TRUNCATION / FIDESLIB_KEY_LEVEL_MARGIN take precedence).
	cc->truncate_keys	 = truncateKeys;
	cc->key_level_margin = 1;

	auto keyPair = cc->KeyGen();
	cc->EvalMultKeyGen(keyPair.secretKey);
	cc->EvalBootstrapSetup(levelBudget, { 0, 0 }, numSlots, 0);
	cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
	cc->LoadContext(keyPair.publicKey);

	std::mt19937 gen(12345); // fixed seed: both runs bootstrap the same vector
	std::uniform_real_distribution<> dis(-1.0, 1.0);
	std::vector<double> x(numSlots);
	for (auto& v : x)
		v = dis(gen);

	Plaintext ptxt = cc->MakeCKKSPackedPlaintext(x, 1, depth - 1, nullptr, numSlots);
	ptxt->SetLength(numSlots);
	Ciphertext<DCRTPoly> ct = cc->Encrypt(keyPair.publicKey, ptxt);

	RunResult r;
	Ciphertext<DCRTPoly> out;
	for (int it = 0; it < repeat; ++it) { // repeat: the first run includes any on-demand key growth.
		auto t0 = std::chrono::high_resolution_clock::now();
		out		= cc->EvalBootstrap(ct);
		auto t1 = std::chrono::high_resolution_clock::now();
		r.bootstrapSec = std::chrono::duration<double>(t1 - t0).count();
	}
	r.levelsAfter = depth - out->GetLevel();

	Plaintext result;
	cc->Decrypt(keyPair.secretKey, out, &result);
	result->SetLength(numSlots);
	auto y = result->GetRealPackedValue();
	for (uint32_t i = 0; i < numSlots; ++i)
		r.maxAbsError = std::max(r.maxAbsError, std::fabs(y[i] - x[i]));

	r.keyBytes	= cc->GetKeyDeviceBytes();
	r.grownKeys = cc->GetGrownKeyCount();
	return r;
}

int main(int argc, char* argv[]) {
	int logN				 = argc > 1 ? std::atoi(argv[1]) : 13;
	uint32_t numSlots		 = argc > 2 ? (1u << std::atoi(argv[2])) : (1u << (logN - 1));
	uint32_t depth			 = argc > 3 ? std::atoi(argv[3]) : 25;
	uint32_t dnum			 = argc > 4 ? std::atoi(argv[4]) : 3;
	uint32_t lb_e			 = argc > 5 ? std::atoi(argv[5]) : 3;
	uint32_t lb_d			 = argc > 6 ? std::atoi(argv[6]) : 3;
	const int repeat		 = 2;

	std::cout << "logN=" << logN << " slots=" << numSlots << " depth=" << depth << " dnum=" << dnum << " levelBudget={" << lb_e << "," << lb_d << "}\n";

	std::cout << "\n=== truncated keys ===\n";
	RunResult trunc = RunBootstrap(true, logN, numSlots, depth, dnum, { lb_e, lb_d }, repeat);
	std::cout << "\n=== complete keys ===\n";
	RunResult full = RunBootstrap(false, logN, numSlots, depth, dnum, { lb_e, lb_d }, repeat);

	auto mib = [](size_t b) { return (double)b / (1 << 20); };
	std::cout << "\n---------------- summary ----------------\n";
	std::cout << "levels after bootstrap : " << trunc.levelsAfter << " (full: " << full.levelsAfter << ")\n";
	std::cout << "max |err|  truncated   : " << trunc.maxAbsError << "\n";
	std::cout << "max |err|  complete    : " << full.maxAbsError << "\n";
	std::cout << "key memory truncated   : " << mib(trunc.keyBytes) << " MiB\n";
	std::cout << "key memory complete    : " << mib(full.keyBytes) << " MiB  (saving " << 100.0 * (1.0 - (double)trunc.keyBytes / (double)std::max<size_t>(1, full.keyBytes)) << " %)\n";
	std::cout << "keys grown at runtime  : " << trunc.grownKeys << "\n";
	std::cout << "bootstrap time (2nd run): truncated " << trunc.bootstrapSec << " s, complete " << full.bootstrapSec << " s\n";

	int rc = 0;
	// The two runs use different random keys, so errors differ by noise; a factor-8 gap means something is wrong.
	if (trunc.maxAbsError > 8.0 * full.maxAbsError + 1e-9) {
		std::cerr << "FAIL: truncated run is significantly less precise\n";
		rc = 1;
	}
	if (trunc.grownKeys > 0) {
		std::cerr << "FAIL: " << trunc.grownKeys << " truncated keys had to be grown at runtime (level plan too tight)\n";
		rc = 1;
	}
	if (trunc.keyBytes >= full.keyBytes) {
		std::cerr << "FAIL: truncation did not reduce key memory\n";
		rc = 1;
	}
	std::cout << (rc == 0 ? "PASS" : "FAIL") << std::endl;
	return rc;
}
