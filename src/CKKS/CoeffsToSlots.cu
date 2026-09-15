//
// Created by carlosad on 27/11/24.
//

#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/Ciphertext.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/LinearTransform.cuh"
#include "CKKS/Plaintext.cuh"
#include <iostream>
#include <ranges>
#include <vector>

#if defined(__clang__)
#include <experimental/source_location>
using sc = std::experimental::source_location;
#else
#include <source_location>
using sc = std::source_location;
#endif

using namespace FIDESlib::CKKS;

constexpr bool BATCHED = false;

void FIDESlib::CKKS::EvalLinearTransform(Ciphertext& ctxt, int slots, bool decode) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	//constexpr bool PRINT		 = false;
	//FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc				 = ctxt.cc;

	if constexpr (BATCHED) {
		/*
		CiphertextBatch<Ciphertext*> bctxt = {.cts = {&ctxt},
											  .conf = {.cc_ = ctxt.cc_,
													   .dims = {{.size = 1}},
													   .level = ctxt.getLevel(),
													   .scale_degree = ctxt.NoiseLevel,
													   .isExt = ctxt.c0.isModUp()}};

		auto& LTconf = cc.GetBootPrecomputation(slots).LT;
		PlaintextBatch<Plaintext*> bptxt = {
			.conf = {.cc_ = ctxt.cc_,
					 .dims = {{.size = 1},
							  {.size = (LTconf.slots + LTconf.bStep - 1) / LTconf.bStep},
							  {.size = LTconf.bStep}},
					 .level = ctxt.getLevel(),
					 .scale_degree = ctxt.NoiseLevel,
					 .isExt = (decode ? LTconf.invA : LTconf.A)[0].c0.isModUp()}};
		for (auto& i : (decode ? LTconf.invA : LTconf.A)) {
			bptxt.cts.push_back(&i);
		}
		int rowSize_padded = bptxt.conf.dims[1].size * bptxt.conf.dims[2].size;
		while (bptxt.cts.size() < rowSize_padded)
			bptxt.cts.push_back(nullptr);

		LinearTransform(bctxt, rowSize_padded, LTconf.bStep, bptxt, 1, 0);
		*/
	} else {

		int bStep				  = cc.GetBootPrecomputation(slots).LT.bStep;
		int gStep				  = slots / bStep;
		std::vector<Plaintext>& A = decode ? cc.GetBootPrecomputation(slots).LT.invA : cc.GetBootPrecomputation(slots).LT.A;
		std::vector<Plaintext*> Aptr(slots, nullptr);
		for (uint32_t j = 0; j < static_cast<uint32_t>(gStep); ++j) {
			for (uint32_t i = 0; i < static_cast<uint32_t>(bStep); ++i) {
				if (bStep * j + i < static_cast<uint32_t>(slots))
					Aptr[bStep * j + i] = &(A[bStep * j + i]);
			}
		}
		LinearTransform(ctxt, slots, bStep, Aptr, 1, 0);
	}
}

void FIDESlib::CKKS::EvalCoeffsToSlots(Ciphertext& ctxt, int slots, bool decode) {
	CudaNvtxRange r(std::string{ sc::current().function_name() });
	constexpr bool PRINT = false;
	// FIDESlib::CKKS::Context& cc_ = ctxt.cc_;
	ContextData& cc = ctxt.cc;

	if constexpr (PRINT) {
		cudaDeviceSynchronize();
		std::cout << "Input stc ";
		for (auto& j : ctxt.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb) {
				SWITCH(i, printThisLimb(1));
			}
		}
		std::cout << std::endl;
		cudaDeviceSynchronize();
	}
	//  No need for Encrypted Bit Reverse
	// Ciphertext& result = ctxt;
	// hoisted automorphisms
	if (ctxt.NoiseLevel == 2)
		ctxt.rescale();

	auto& steps = decode ? cc.GetBootPrecomputation(slots).StC : cc.GetBootPrecomputation(slots).CtS;

	// The diagonals come from OpenFHE's EvalBootstrapSetup, which encodes them at a level of its own
	// choosing, while ModRaise here grows the ciphertext to `cc.L` (minus one only for
	// FLEXIBLEAUTOEXT). Under FIXEDMANUAL those disagree by a limb, and the batched product does not
	// tolerate that the way OpenFHE's EvalMult(ct, pt) does - it sizes the launch from the ciphertext
	// and indexes the plaintext with the same bound, so the kernel reads one limb past the end.
	// Observed on a GV100 as "would read 35 limbs from pt[0], which holds 34".
	//
	// Drop the ciphertext to the diagonals of the step about to run, which is what OpenFHE's
	// AdjustLevelsAndDepth does implicitly. Done per step rather than once, because each layer has its
	// own diagonals and its own level.
	//
	// FIXEDMANUAL only. The other techniques adjust levels themselves and their diagonals are encoded
	// to match, so there is nothing to align - and forcing the drop anyway takes levels the caller
	// still needs. That is not hypothetical: unconditional, this drove a FLEXIBLEAUTO bootstrap's
	// level negative and the example crashed indexing RNSLimbs[-1], on a branch whose only difference
	// from a working one was this function.
	const bool alignNeeded = cc.rescaleTechnique == FIXEDMANUAL;
	//
	// Taking the *minimum* level assumes every diagonal of a step is encoded at the same one. If they
	// are not, dropping to the lowest leaves the higher ones truncated by `multPt`, which sizes the
	// plaintext read from the ciphertext - the limbs it cuts carry that diagonal's residues under the
	// primes it dropped. The value survives, but it comes out wrong, and wrong differently per slot
	// because each diagonal feeds different slots. That is a candidate for the bootstrap's measured
	// 10.5-bit accuracy (bugs/RESPONSE-bootstrap-precision-cts-20260915.md), so the assumption is
	// reported rather than assumed. Once per process, and only when it does not hold.
	static bool reportedUnevenDiagonals = false;
	const auto alignToDiagonals = [&alignNeeded](Ciphertext& ct, const BootstrapPrecomputation::LTstep& step) {
		if (!alignNeeded)
			return;
		int ptLevel = -1, ptLevelMax = -1;
		for (const Plaintext& pt : step.A) {
			const int level = pt.c0.getLevel();
			if (level < 0)
				continue;
			if (ptLevel < 0 || level < ptLevel)
				ptLevel = level;
			if (level > ptLevelMax)
				ptLevelMax = level;
		}
		if (ptLevel >= 0 && ptLevelMax > ptLevel && !reportedUnevenDiagonals) {
			reportedUnevenDiagonals = true;
			std::cerr << "FIDESlib: the diagonals of a bootstrap linear-transform step are not all at "
					  << "one level (" << ptLevel << " to " << ptLevelMax << ", " << step.A.size()
					  << " diagonals). Aligning the ciphertext to the lowest truncates the rest, which "
					  << "decrypts to a wrong value rather than failing. See "
					  << "bugs/RESPONSE-bootstrap-precision-cts-20260915.md." << std::endl;
		}
		if (ptLevel >= 0 && ct.getLevel() > ptLevel)
			ct.dropToLevel(ptLevel);
	};

	for (BootstrapPrecomputation::LTstep& step : steps) {
		alignToDiagonals(ctxt, step);
		// computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)

		if constexpr (BATCHED) {
			/*
			CiphertextBatch<Ciphertext*> bctxt = {.cts = {&ctxt},
												  .conf = {.cc_ = ctxt.cc_,
														   .dims = {{.size = 1}},
														   .level = ctxt.getLevel(),
														   .scale_degree = ctxt.NoiseLevel,
														   .isExt = ctxt.c0.isModUp()}};

			if (bctxt.conf.scale_degree == 2)
				bctxt.Rescale();

			PlaintextBatch<Plaintext*> bptxt = {
				.conf = {
					.cc_ = ctxt.cc_,
					.dims = {{.size = 1}, {.size = (step.slots + step.bStep - 1) / step.bStep}, {.size = step.bStep}},
					.level = ctxt.getLevel(),
					.scale_degree = ctxt.NoiseLevel,
					.isExt = step.A[0].c0.isModUp()}};
			for (auto& i : step.A) {
				bptxt.cts.push_back(&i);
			}
			int rowSize_padded = bptxt.conf.dims[1].size * bptxt.conf.dims[2].size;
			while (bptxt.cts.size() < rowSize_padded)
				bptxt.cts.push_back(nullptr);

			LinearTransform(bctxt, rowSize_padded, step.bStep, bptxt, step.rotIn[1] - step.rotIn[0], step.rotOut[0]);
			*/
		} else {

			{

				assert(step.slots == step.A.size());
				std::vector<Plaintext*> Aptr(step.slots, nullptr);
				for (int j = 0; j < step.gStep; ++j) {
					for (int i = 0; i < step.bStep; ++i) {
						if (step.bStep * j + i < step.slots)
							Aptr[step.bStep * j + i] = &(step.A[step.bStep * j + i]);
					}
				}

				int stride = step.bStep > 1 ? step.rotIn[1] - step.rotIn[0] : step.rotOut[1];
				int offset = step.rotOut[0];
				{
					LinearTransform(ctxt, step.slots, step.bStep, Aptr, stride, offset);
				}
			}
		}
	}
}
