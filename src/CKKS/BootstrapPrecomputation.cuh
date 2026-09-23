//
// Created by carlosad on 27/11/24.
//

#ifndef GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
#define GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH

#define AFFINE_LT true

#include "Plaintext.cuh"
#include <vector>

namespace FIDESlib::CKKS {

class BootstrapPrecomputation {
  public:
	struct {
		int slots = -1;
		int bStep = -1;
		std::vector<Plaintext> A;
		std::vector<Plaintext> invA;
	} LT;

	struct LTstep {
		int slots = -1;
		int bStep = -1;
		int gStep = -1;
		std::vector<Plaintext> A;
		std::vector<int> rotIn;
		std::vector<int> rotOut;
	};

	std::vector<LTstep> StC;
	std::vector<LTstep> CtS;
	int accumulate_bStep = 4;
	uint32_t correctionFactor;
	bool sparse_encaps{ false };
	std::weak_ptr<ContextData> sparse_context;

	/// @brief What a bootstrap built from this precomputation promises to hand back.
	///
	/// A bootstrap is the one operation whose output state a caller cannot derive: how many levels
	/// it leaves depends on the level budget, the secret key distribution and the level the
	/// precomputed diagonals were encoded at. When that state is merely whatever came out, a
	/// deviation is invisible until it surfaces somewhere else entirely - which has now happened
	/// twice here. `EvalBootstrap` returned scale degree 2 under FIXEDMANUAL (72dc818), where the
	/// degree only ever grows on multiply and falls on rescale, so starting at 2 reached 12108 by
	/// the end of a softmax and the first addPt failed on mismatched scales; before that it was
	/// silent. The correction was a rescale loop in the Python wrapper, far from the operation that
	/// broke the contract and costing a level nobody accounted for.
	///
	/// So the contract is declared rather than observed: `noiseLevel` is 1 because that is what
	/// FIXEDMANUAL means and what `approxModReduction` already rescales to achieve, and the
	/// bootstrap is held to it at its own exit. `level` is pinned on the first bootstrap and
	/// checked against every later one, which catches drift without having to predict a number
	/// that depends on three separate parameters.
	struct OutputContract {
		int noiseLevel = 1;      ///< declared: canonical scale degree under FIXEDMANUAL
		int level      = -1;     ///< pinned on the first bootstrap, then enforced
		bool levelPinned = false;
	};
	OutputContract output;

	/// @brief Identity of the context these precomputations were built against.
	///
	/// Precomputations outlive the call that made them and are looked up by slot count alone, so
	/// nothing otherwise stops a context whose parameters have moved from finding and using them.
	/// The diagonals would be encoded for the wrong modulus chain and the failure would look
	/// numerical.
	struct Fingerprint {
		int logN = -1;
		int L = -1;
		int dnum = -1;
		int K = -1;
		int rescaleTechnique = -1;

		bool valid() const { return logN >= 0; }
		bool operator==(const Fingerprint& o) const {
			return logN == o.logN && L == o.L && dnum == o.dnum && K == o.K &&
				   rescaleTechnique == o.rescaleTechnique;
		}
		bool operator!=(const Fingerprint& o) const { return !(*this == o); }
	};
	Fingerprint fingerprint;
};

} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAPPRECOMPUTATION_CUH
