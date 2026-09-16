//
// Created by carlosad on 4/12/24.
//

#ifndef GPUCKKS_BOOTSTRAP_CUH
#define GPUCKKS_BOOTSTRAP_CUH

#include "forwardDefs.cuh"
#include "pke/openfhe.h"

namespace FIDESlib::CKKS {
void BootstrapCPUraise(Ciphertext& ctxt,
  const int slots,
  std::shared_ptr<lbcrypto::CryptoContextImpl<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<expdtype>>>>>& CPUcc,
  lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys,
  const bool prescaled);
// void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false);
/// @brief Refresh `ctxt`. `stopAfterStage` in 1..4 returns after that step instead of finishing, so
/// a caller can decrypt what the bootstrap has built so far; -1 (the default) runs all of it. The
/// intermediate is for measurement only - its level, scale and slot layout are mid-flight.
void Bootstrap(Ciphertext& ctxt, const int slots, const bool prescaled = false,
			   const int stopAfterStage = -1);
double GetPreScaleFactor(Context& cc, int slots);
void ModRaise(Ciphertext& ctxt, const int slots, const uint32_t correction, const bool prescaled = false, bool sparse_encaps = false);
} // namespace FIDESlib::CKKS

#endif // GPUCKKS_BOOTSTRAP_CUH
