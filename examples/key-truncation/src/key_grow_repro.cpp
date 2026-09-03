// Minimal reproducer for the level-truncated rotation key paths, without Python in the loop.
//
// Three scenarios, selected by argv[1]:
//   inside  (default) : key declared for level `plan`, used at level `plan` -> must be exact.
//   above             : key declared for level `plan`, used at the top level with growth disabled
//                       -> must throw a diagnostic (and must NOT crash).
//   grow              : same, with growth enabled -> the key is reloaded and rebuilt, result must be exact.
//
// Usage: key-grow-repro [inside|above|grow|all] [logN=13] [depth=12] [dnum=3] [plan=5] [index=2]
//
// Run under compute-sanitizer to localise device faults, e.g.
//   compute-sanitizer --tool memcheck --launch-timeout 120 ./build-kt/key-grow-repro grow
// Exit code 0 on success.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <fideslib.hpp>

using namespace fideslib;

namespace {

struct Params {
	int logN	= 13;
	uint32_t depth	= 12;
	uint32_t dnum	= 3;
	int plan	= 5; // max remaining levels declared for `index`
	int index	= 2;
};

struct Setup {
	CryptoContext<DCRTPoly> cc;
	KeyPair<DCRTPoly> keys;
	uint32_t slots;
};

Setup Build(const Params& p, bool allowGrow) {
	CCParams<CryptoContextCKKSRNS> parameters;
	parameters.SetSecretKeyDist(UNIFORM_TERNARY);
	parameters.SetSecurityLevel(HEStd_NotSet);
	parameters.SetRingDim(1 << p.logN);
	parameters.SetNumLargeDigits(p.dnum);
	parameters.SetKeySwitchTechnique(HYBRID);
	parameters.SetDevices({ 0 });
	parameters.SetScalingModSize(50);
	parameters.SetFirstModSize(55);
	parameters.SetScalingTechnique(FIXEDMANUAL);
	parameters.SetMultiplicativeDepth(p.depth);

	Setup s;
	s.slots = 1u << (p.logN - 1);
	s.cc	= GenCryptoContext(parameters);
	s.cc->Enable(PKE);
	s.cc->Enable(KEYSWITCH);
	s.cc->Enable(LEVELEDSHE);
	s.cc->Enable(ADVANCEDSHE);

	s.cc->truncate_keys	  = true;
	s.cc->key_level_margin = 0; // no slack: the declared level is exactly the usable level
	s.cc->allow_key_grow   = allowGrow;

	s.keys = s.cc->KeyGen();
	s.cc->EvalMultKeyGen(s.keys.secretKey);
	s.cc->SetRotationKeyLevels({ { p.index, (uint32_t)p.plan } });
	s.cc->EvalRotateKeyGen(s.keys.secretKey, { p.index });
	s.cc->LoadContext(s.keys.publicKey);
	return s;
}

/** Rotate a ciphertext sitting at `remainingLevels` and return the max |error| against the plaintext rotation. */
double RotateAndCheck(Setup& s, const Params& p, uint32_t remainingLevels) {
	std::mt19937 gen(7);
	std::uniform_real_distribution<> dis(-1.0, 1.0);
	std::vector<double> x(s.slots);
	for (auto& v : x)
		v = dis(gen);

	// OpenFHE's `level` argument counts *consumed* levels.
	Plaintext ptxt = s.cc->MakeCKKSPackedPlaintext(x, 1, p.depth - remainingLevels, nullptr, s.slots);
	ptxt->SetLength(s.slots);
	Ciphertext<DCRTPoly> ct = s.cc->Encrypt(s.keys.publicKey, ptxt);
	Ciphertext<DCRTPoly> rot = s.cc->EvalRotate(ct, p.index);

	Plaintext out;
	s.cc->Decrypt(s.keys.secretKey, rot, &out);
	out->SetLength(s.slots);
	auto y = out->GetRealPackedValue();

	double err = 0.0;
	const int n = (int)s.slots;
	for (int i = 0; i < n; ++i)
		err = std::max(err, std::fabs(y[i] - x[((i + p.index) % n + n) % n]));
	return err;
}

bool RunInside(const Params& p) {
	std::cout << "[inside] key for index " << p.index << " declared at level " << p.plan << ", used at level " << p.plan << "\n";
	Setup s	   = Build(p, false);
	double err = RotateAndCheck(s, p, (uint32_t)p.plan);
	std::cout << "  max |err| = " << err << ", key bytes = " << (s.cc->GetKeyDeviceBytes() >> 10) << " KiB, grown = " << s.cc->GetGrownKeyCount() << "\n";
	bool ok = err < 1e-6 && s.cc->GetGrownKeyCount() == 0;
	std::cout << (ok ? "  PASS\n" : "  FAIL\n");
	return ok;
}

bool RunAbove(const Params& p) {
	std::cout << "[above] key declared at level " << p.plan << ", used at level " << p.depth << " with growth disabled\n";
	Setup s = Build(p, false);
	try {
		RotateAndCheck(s, p, p.depth);
	} catch (const std::exception& e) {
		std::cout << "  threw as expected: " << e.what() << "\n  PASS\n";
		return true;
	}
	std::cout << "  FAIL: no exception - a key was used above the level it was loaded for\n";
	return false;
}

bool RunGrow(const Params& p) {
	std::cout << "[grow] key declared at level " << p.plan << ", used at level " << p.depth << " with growth enabled\n";
	Setup s			= Build(p, true);
	size_t before	= s.cc->GetKeyDeviceBytes();
	double err		= RotateAndCheck(s, p, p.depth);
	size_t after	= s.cc->GetKeyDeviceBytes();
	std::cout << "  max |err| = " << err << ", key bytes " << (before >> 10) << " -> " << (after >> 10) << " KiB, grown = " << s.cc->GetGrownKeyCount() << "\n";
	// A second rotation must be served by the now-complete key without another reload.
	double err2 = RotateAndCheck(s, p, p.depth);
	std::cout << "  max |err| (2nd) = " << err2 << ", grown = " << s.cc->GetGrownKeyCount() << "\n";
	bool ok = err < 1e-6 && err2 < 1e-6 && after > before && s.cc->GetGrownKeyCount() == 1;
	std::cout << (ok ? "  PASS\n" : "  FAIL\n");
	return ok;
}

} // namespace

int main(int argc, char* argv[]) {
	std::string what = argc > 1 ? argv[1] : "all";
	Params p;
	if (argc > 2)
		p.logN = std::atoi(argv[2]);
	if (argc > 3)
		p.depth = (uint32_t)std::atoi(argv[3]);
	if (argc > 4)
		p.dnum = (uint32_t)std::atoi(argv[4]);
	if (argc > 5)
		p.plan = std::atoi(argv[5]);
	if (argc > 6)
		p.index = std::atoi(argv[6]);

	std::cout << "logN=" << p.logN << " depth=" << p.depth << " dnum=" << p.dnum << " plan=" << p.plan << " index=" << p.index << "\n";

	bool ok = true;
	if (what == "inside" || what == "all")
		ok &= RunInside(p);
	if (what == "above" || what == "all")
		ok &= RunAbove(p);
	if (what == "grow" || what == "all")
		ok &= RunGrow(p);

	std::cout << (ok ? "ALL PASS" : "FAILURES") << std::endl;
	return ok ? 0 : 1;
}
