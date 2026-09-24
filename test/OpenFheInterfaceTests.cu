//
// Created by carlosad on 29/04/24.
//

#include <openfhe.h>
#undef duration
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/Limb.cuh"
#include "CKKS/Plaintext.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"
#include "ConstantsGPU.cuh"
#include "Math.cuh"
#include "ParametrizedTest.cuh"
#include "cpuNTT.hpp"
#include "cpuNTT_nega.hpp"
#include <CKKS/AccumulateBroadcast.cuh>
#include <gtest/gtest.h>
#include <iomanip>
// #include "hook.h"
#include "CKKS/ApproxModEval.cuh"
#include "CKKS/Bootstrap.cuh"
#include "CKKS/BootstrapPrecomputation.cuh"
#include "CKKS/CoeffsToSlots.cuh"
#include "CKKS/openfhe-interface/ParameterSwitch.cuh"

namespace FIDESlib::Testing {
class OpenFHEInterfaceTest : public GeneralParametrizedTest {
};

TEST_P(OpenFHEInterfaceTest, ExtractContextShowAdd) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	// FIDESlib::Constants& host_constants = FIDESlib::CKKS::GetCurrentContext()->precom.constants[0];
	// FIDESlib::Global& host_global = *FIDESlib::CKKS::GetCurrentContext()->precom.globals;

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(cc_, raw2);

	// GPU add
	GPUct1.add(GPUct2);
	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c3);

	GetOpenFHECipherText(cResGPU, raw_res1);
	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	// CPU add
	lbcrypto::Plaintext result;
	auto cAdd = cc->EvalAdd(c1, c2);
	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Add:\n";
	std::cout << "Result " << result;
	// result2->SetLength(batchSize);
	std::cout << "Result GPU " << resultGPU;

	ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

	FIDESlib::CKKS::RawPlainText rawpt = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);

	// CPU addPt
	auto cAddPt = cc->EvalAdd(cAdd, ptxt1);
	cc->Decrypt(keys.secretKey, cAddPt, &result);

	// GPU addPt
	FIDESlib::CKKS::Plaintext GPUpt1(cc_, rawpt);
	GPUct1.addPt(GPUpt1);
	GPUct1.store(raw_res1);

	GetOpenFHECipherText(cResGPU, raw_res1);
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	std::cout << "AddPt:\n";
	std::cout << "Result " << result;
	// result2->SetLength(batchSize);
	std::cout << "Result GPU " << resultGPU;

	ASSERT_EQ_CIPHERTEXT(cAddPt, cResGPU);
}

TEST_P(OpenFHEInterfaceTest, ExtractContextPtAutomorph) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	// FIDESlib::Constants& host_constants = FIDESlib::CKKS::GetCurrentContext()->precom.constants[0];
	// FIDESlib::Global& host_global = *FIDESlib::CKKS::GetCurrentContext()->precom.globals;

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(cc_, raw2);

	// GPU add
	GPUct1.add(GPUct2);
	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c3);

	GetOpenFHECipherText(cResGPU, raw_res1);
	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	// CPU add
	lbcrypto::Plaintext result;
	auto cAdd = cc->EvalAdd(c1, c2);
	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Add:\n";
	std::cout << "Result " << result;
	// result2->SetLength(batchSize);
	std::cout << "Result GPU " << resultGPU;

	ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

	FIDESlib::CKKS::RawPlainText rawpt = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);

	// CPU addPt
	auto cAddPt = cc->EvalAdd(cAdd, ptxt1);
	cc->Decrypt(keys.secretKey, cAddPt, &result);

	// GPU addPt
	FIDESlib::CKKS::Plaintext GPUpt1(cc_, rawpt);
	GPUpt1.automorph(1);
	GPUct1.addPt(GPUpt1);
	GPUct1.store(raw_res1);

	GetOpenFHECipherText(cResGPU, raw_res1);
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	std::cout << "AddPt:\n";
	std::cout << "Result " << result;
	// result2->SetLength(batchSize);
	std::cout << "Result GPU " << resultGPU;
}

TEST_P(OpenFHEInterfaceTest, ExtractContextCreateSwitch) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	cc->GetEncodingParams()->SetBatchSize(8);

	auto cc_switch = CKKS::createSwitchableContextBasedOnContext(cc, 1, 1, cc->GetRingDimension() / 2);

	auto [swtch, sk_sparse] = CKKS::createContextSwitchingKeys(cc, cc_switch, keys, 32);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	FIDESlib::CKKS::RawParams raw_param2 = FIDESlib::CKKS::GetRawParams(cc_switch);
	FIDESlib::CKKS::Context cc_switch_   = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param2), devices);
	FIDESlib::CKKS::ContextData& GPUcc2  = *cc_switch_;

	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetKeySwitchKey(swtch.first);
	FIDESlib::CKKS::KeySwitchingKey ksk_atob(cc_switch_);
	ksk_atob.Initialize(rawKskEval);

	FIDESlib::CKKS::RawKeySwitchKey rawKskEval2 = FIDESlib::CKKS::GetKeySwitchKey(swtch.second);
	FIDESlib::CKKS::KeySwitchingKey ksk_btoa(cc_);
	ksk_btoa.Initialize(rawKskEval2);

	CKKS::AddSecretSwitchingKey(std::move(ksk_atob), std::move(ksk_btoa));

	auto& atob = CKKS::GetSecretSwitchingKey(cc_, cc_switch_, keys.publicKey->GetKeyTag());
	auto& btoa = CKKS::GetSecretSwitchingKey(cc_switch_, cc_, keys.publicKey->GetKeyTag());

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

	FIDESlib::CKKS::Ciphertext GPUct_switched(cc_switch_);

	{
		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c1);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;
	}

	if (GPUct1.NoiseLevel == 2)
		GPUct1.rescale();
	GPUct1.dropToLevel(cc_switch_->L - cc_switch_->rescaleTechnique == CKKS::FLEXIBLEAUTOEXT);
	// std::cout << "Reinterpret 1 " << std::endl;

	GPUct_switched.reinterpretContext(GPUct1);
	// std::cout << "KS 1 " << std::endl;
	GPUct_switched.keySwitch(atob);

	if (0) {
		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct_switched.store(raw_res1);
		auto cResGPU(c1);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc_switch->Decrypt(sk_sparse, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;
	}

	// std::cout << "Add " << std::endl;

	GPUct_switched.add(GPUct_switched);

	// std::cout << "Reinterpret 2 " << std::endl;

	GPUct1.reinterpretContext(GPUct_switched);
	// std::cout << "Switch 2 " << std::endl;

	GPUct1.keySwitch(btoa);
	// std::cout << "End " << std::endl;

	{
		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c1);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;
	}
}

TEST_P(OpenFHEInterfaceTest, ScalarAdd) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

	FIDESlib::CKKS::Ciphertext GPUct2(cc_, raw1);

	// GPU add
	GPUct1.addScalar(2.0);
	GPUct2.addScalar(-2.0);
	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c3);

	GetOpenFHECipherText(cResGPU, raw_res1);
	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	// CPU add
	lbcrypto::Plaintext result;
	auto cAdd = cc->EvalAdd(c1, 2.0);

	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Add:\n";
	std::cout << "Result " << result;
	// result2->SetLength(batchSize);
	std::cout << "Result GPU " << resultGPU;

	auto cSub = cc->EvalAdd(c1, -2.0);

	cc->Decrypt(keys.secretKey, cSub, &result);

	FIDESlib::CKKS::RawCipherText raw_res2;
	GPUct2.store(raw_res2);
	auto cResGPU2 = c3->Clone();
	GetOpenFHECipherText(cResGPU2, raw_res2);
	lbcrypto::Plaintext resultGPU2;
	cc->Decrypt(keys.secretKey, cResGPU2, &resultGPU2);
	std::cout << "Sub:\n";
	std::cout << "Result " << result;
	// result2->SetLength(batchSize);
	std::cout << "Result GPU " << resultGPU2;

	ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);
	ASSERT_EQ_CIPHERTEXT(cSub, cResGPU2);
}

TEST_P(OpenFHEInterfaceTest, ExtractContextRunNTT) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText raw2  = FIDESlib::CKKS::GetRawPlainText(cc, ptxt2);

	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(cc_, raw1);
	FIDESlib::CKKS::Plaintext GPUpt2(cc_, raw2);

	c1->GetElements().at(0) /*m_elements[0]*/.SwitchFormat();
	// c1->m_elements[0].SwitchFormat();

	GPUct1.c0.INTT<ALGO_NATIVE>(1, false);
	// GPUct1.c0.NTT();

	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c3);
	GetOpenFHECipherText(cResGPU, raw_res1);

	/*
		for(auto & k :c1->m_elements[0].m_vectors[0].m_values->m_data){
			k.m_value = FIDESlib::modprod(k.m_value, FIDESlib::host_constants.N, FIDESlib::host_constants.primes[0]);
		}
		 */

	/*
			std::vector<uint64_t> v;
			for(auto & k :cResGPU->m_elements[0].m_vectors[0].m_values->m_data){
				v.push_back(k.m_value);
			}
			FIDESlib::bit_reverse_vector(v);
			FIDESlib::nega_fft2_forPrime(v, true, 0);
			for(int k = 0; k < v.size(); ++k){
				cResGPU->m_elements[0].m_vectors[0].m_values->m_data[k].m_value = v[k];
			}
		*/

	ASSERT_EQ_CIPHERTEXT(c1, cResGPU);
	/*
	for (int j = 0; j < 1; ++j) {
		ASSERT_EQ(c1.get()->m_elements[j].m_vectors.size(), cResGPU.get()->m_elements[j].m_vectors.size());

		for (size_t i = 0; i < c1.get()->m_elements[j].m_vectors.size(); ++i) {
			// std::cout << "i = " << i << ", j = " << j << std::endl;
			ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_params->m_ciphertextModulus, GPUcc.prime[i].p);

			ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.size(),
					  cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.size());

			for (int k = 0; k < GPUcc.N; ++k)
				if (c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data[k] !=
					cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data[k]) {
					std::cout << std::hex << i << ":" << k << " "
							  << c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data[k] << " "
							  << cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data[k] << std::endl;
				}

			//std::sort(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.begin(),c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.end() );
			//std::sort(cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.begin(),cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data.end() );

			ASSERT_EQ(c1.get()->m_elements[j].m_vectors[i].m_values.get()->m_data,
					  cResGPU.get()->m_elements[j].m_vectors[i].m_values.get()->m_data);
		}
	}
	*/
}

TEST_P(OpenFHEInterfaceTest, ExtractContextShowPtMult) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	FIDESlib::Constants& host_constants = FIDESlib::CKKS::GetCurrentContext()->precom.constants[0];
	FIDESlib::Global& host_global       = *FIDESlib::CKKS::GetCurrentContext()->precom.globals;

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText raw2  = FIDESlib::CKKS::GetRawPlainText(cc, ptxt2);

	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2_(cc_, raw1);
	FIDESlib::CKKS::Plaintext GPUpt2(cc_, raw2);

	// CPU ptMult

	auto cMultNoRes = cc->EvalMult(c1, ptxt2);
	// auto cAux = cc->EvalMult(c1, c1);
	lbcrypto::Plaintext result;
	cc->Decrypt(keys.secretKey, cMultNoRes, &result);

	std::cout << "MultPt:\n";
	std::cout << "Result " << result;

	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cMult;
	if (GPUcc.rescaleTechnique == CKKS::FIXEDMANUAL) {
		cMult = cc->Rescale(cMultNoRes);
	} else {
		cMult = cc->EvalMult(cMultNoRes, ptxt2);
	}

	cc->Decrypt(keys.secretKey, cMult, &result);

	std::cout << "MultPt:\n";
	std::cout << "Result " << result;

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		FIDESlib::CKKS::Ciphertext GPUct2(cc_);
		GPUct1.copy(GPUct1_);
		GPUct2.copy(GPUct2_);

		// GPU ptMult
		GPUct1.multPt(GPUpt2, false);
		GPUct2.multPt(GPUpt2, true);
		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU(c3->Clone());
			GetOpenFHECipherText(cResGPU, raw_res1);
			lbcrypto::Plaintext resultGPU;

			ASSERT_EQ_CIPHERTEXT(cResGPU, cMultNoRes);
			cc->RescaleInPlace(cResGPU);
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

			std::cout << "Result GPU with OpenFHE rescale " << resultGPU;
		}

		if (GPUcc.rescaleTechnique == CKKS::FIXEDMANUAL) {
			GPUct1.rescale();
		} else {
			GPUct1.multPt(GPUpt2, false);
			GPUct2.multPt(GPUpt2, false);
		}

		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU(c3->Clone());
			GetOpenFHECipherText(cResGPU, raw_res1);

			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU with rescale " << resultGPU;
			ASSERT_EQ_CIPHERTEXT(cMult, cResGPU);
			{
				const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
				for (int i = 0; i < GPUcc.L; ++i) {
					for (int j = 0; j <= GPUcc.L; ++j) {
						if (i < j) {
							ASSERT_EQ(host_global.q_inv[j][i], cryptoParams->GetqlInvModq(GPUcc.L - j)[i]);
						}
					}
				}

				for (int i = 0; i < GPUcc.L; ++i) {
					for (int j = 0; j <= GPUcc.L; ++j) {
						if (i < j) {
							ASSERT_EQ(host_global.QlQlInvModqlDivqlModq[j][i], cryptoParams->GetQlQlInvModqlDivqlModq(GPUcc.L - j)[i]);
						}
					}
				}
			}

			FIDESlib::CKKS::RawCipherText raw_res2;
			GPUct2.store(raw_res2);
			auto cResGPU2(c3->Clone());
			GetOpenFHECipherText(cResGPU2, raw_res2);

			lbcrypto::Plaintext resultGPU2;
			cc->Decrypt(keys.secretKey, cResGPU2, &resultGPU2);

			std::cout << "Result GPU with fused ptmult" << resultGPU2;

			ASSERT_EQ_CIPHERTEXT(cMult, cResGPU2);
		}
	}
}

/*
TEST_P(OpenFHEInterfaceTest, ExtractContextShowPtMultSquareScale) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_		= GPUcc;
	cc_									= CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc	= *cc_;
	FIDESlib::Constants& host_constants = FIDESlib::CKKS::GetCurrentContext()->precom.constants[0];
	FIDESlib::Global& host_global		= *FIDESlib::CKKS::GetCurrentContext()->precom.globals;

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText raw2  = FIDESlib::CKKS::GetRawPlainText(cc, ptxt2);

	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2_(cc_, raw1);
	FIDESlib::CKKS::Plaintext GPUpt2(cc_, raw2);

	// CPU ptMult

	auto cMultNoRes = cc->EvalMult(c1, ptxt2);
	// auto cAux = cc->EvalMult(c1, c1);
	lbcrypto::Plaintext result;
	cc->Decrypt(keys.secretKey, cMultNoRes, &result);

	std::cout << "MultPt:\n";
	std::cout << "Result " << result;

	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cMult;
	if (GPUcc.rescaleTechnique == CKKS::FIXEDMANUAL) {
		cMult = cc->Rescale(cMultNoRes);
	} else {
		cMult = cc->EvalMult(cMultNoRes, ptxt2);
	}

	cc->Decrypt(keys.secretKey, cMult, &result);

	std::cout << "MultPt:\n";
	std::cout << "Result " << result;

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		FIDESlib::CKKS::Ciphertext GPUct2(cc_);
		GPUct1.copy(GPUct1_);
		GPUct2.copy(GPUct2_);

		// GPU ptMult
		if (GPUpt2.NoiseLevel == 2)
			GPUpt2.rescale();
		GPUct1.multPt(GPUpt2, false);
		GPUct1.multPt(GPUpt2, false, true);
		GPUct1.rescale();
		GPUct1.rescale();
		GPUct2.multPt(GPUpt2, true);
		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU(c3->Clone());
			GetOpenFHECipherText(cResGPU, raw_res1);
			lbcrypto::Plaintext resultGPU;

			// ASSERT_EQ_CIPHERTEXT(cResGPU, cMultNoRes);
			// cc->RescaleInPlace(cResGPU);
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

			std::cout << "Result GPU with OpenFHE rescale " << resultGPU;
		}

		if (GPUcc.rescaleTechnique == CKKS::FIXEDMANUAL) {
			GPUct1.rescale();
		} else {
			GPUct1.multPt(GPUpt2, false);
			GPUct2.multPt(GPUpt2, false);
		}

		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU(c3->Clone());
			GetOpenFHECipherText(cResGPU, raw_res1);

			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU with rescale " << resultGPU;
			ASSERT_EQ_CIPHERTEXT(cMult, cResGPU);
			{
				const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
				for (int i = 0; i < GPUcc.L; ++i) {
					for (int j = 0; j <= GPUcc.L; ++j) {
						if (i < j) {
							ASSERT_EQ(host_global.q_inv[j][i], cryptoParams->GetqlInvModq(GPUcc.L - j)[i]);
						}
					}
				}

				for (int i = 0; i < GPUcc.L; ++i) {
					for (int j = 0; j <= GPUcc.L; ++j) {
						if (i < j) {
							ASSERT_EQ(host_global.QlQlInvModqlDivqlModq[j][i], cryptoParams->GetQlQlInvModqlDivqlModq(GPUcc.L - j)[i]);
						}
					}
				}
			}

			FIDESlib::CKKS::RawCipherText raw_res2;
			GPUct2.store(raw_res2);
			auto cResGPU2(c3->Clone());
			GetOpenFHECipherText(cResGPU2, raw_res2);

			lbcrypto::Plaintext resultGPU2;
			cc->Decrypt(keys.secretKey, cResGPU2, &resultGPU2);

			std::cout << "Result GPU with fused ptmult" << resultGPU2;

			ASSERT_EQ_CIPHERTEXT(cMult, cResGPU2);
		}
	}
}
*/
TEST_P(OpenFHEInterfaceTest, InitializeOpenFHE) {

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	cc->EvalMultKeyGen(keys.secretKey);

	cc->EvalRotateKeyGen(keys.secretKey, { 1, -2 });

	// Step 3: Encoding and encryption of inputs

	// Inputs
	// vector of c1 and c2, for loop running of evalAdd over vectors
	// will need to make it multithreaded

	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 5.0, 4.0, 3.0, 2.0, 1.0, 0.75, 0.5, 0.25 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	auto cAdd = cc->EvalAdd(c1, c2);
	lbcrypto::Plaintext result;
	std::cout.precision(8);
	std::cout << std::endl << "Results of homomorphic computations: " << std::endl;

	cc->Decrypt(keys.secretKey, cAdd, &result);
	result->SetLength(generalTestParams.batchSize);
	std::cout << "x1 = " << result;
	std::cout << "Estimated precision in bits: " << result->GetLogPrecision() << std::endl;
}

#if MODRAISE_WITH_P0
lbcrypto::Plaintext
encodeExt(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, const std::vector<double>& value, size_t noiseScaleDeg, uint32_t L, uint32_t K, int slots) {

	uint32_t M              = cc->GetCyclotomicOrder();
	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

	lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer> elementParams = *(cryptoParams->GetElementParams());

	uint32_t towersToDrop = 0;

	if (L != 0) {
		towersToDrop = elementParams.GetParams().size() - L - 1;
	}
	for (uint32_t i = 0; i < towersToDrop; i++) {
		elementParams.PopLastParam();
	}

	auto paramsQ   = elementParams.GetParams();
	uint32_t sizeQ = paramsQ.size();
	auto paramsP   = cryptoParams->GetParamsP()->GetParams();
	{
		uint32_t towersToDrop = 0;
		if (K != 0) {
			towersToDrop = paramsP.size() - K;
		}
		for (uint32_t i = 0; i < towersToDrop; i++) {
			paramsP.pop_back();
		}
	}
	uint32_t sizeP = paramsP.size();
	std::vector<NativeInteger> moduli(sizeQ + sizeP);
	std::vector<NativeInteger> roots(sizeQ + sizeP);
	for (size_t i = 0; i < sizeQ; i++) {
		moduli[i] = paramsQ[i]->GetModulus();
		roots[i]  = paramsQ[i]->GetRootOfUnity();
	}

	for (size_t i = 0; i < sizeP; i++) {
		moduli[sizeQ + i] = paramsP[i]->GetModulus();
		roots[sizeQ + i]  = paramsP[i]->GetRootOfUnity();
	}

	auto elementParamsPtr = std::make_shared<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>>(M, moduli, roots);

	// auto res = cc->MakeCKKSPackedPlaintext(value, noiseScaleDeg, L, elementParamsPtr);

	// return res;
	std::vector<std::complex<double>> v;
	std::ranges::transform(value, std::back_inserter(v), [](double r) { return std::complex<double>(r, 0); });

	return std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)
		->MakeAuxPlaintext(*cc,
		                   elementParamsPtr,
		                   v,
		                   noiseScaleDeg,
		                   towersToDrop,
		                   slots
		                   //	,
		                   //	(noiseScaleDeg == 2 && K > 0) ?
		                   //	  sqrt(cryptoParams->GetScalingFactorReal(cryptoParams->GetScalingTechnique() == lbcrypto::FLEXIBLEAUTOEXT) * moduli.back().ConvertToDouble()) : 0
		);
}

TEST_P(OpenFHEInterfaceTest, Rescale) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	if constexpr (MODRAISE_WITH_P0) {
		// Encoding as plaintexts
		// lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 2, 1);
		lbcrypto::Plaintext ptxt1 = encodeExt(cc, x1, 2, 0, 1, 8);
		lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

		std::cout << "Input x1: " << ptxt1 << std::endl;

		// Encrypt the encoded vectors
		auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
		auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
		FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

		lbcrypto::Plaintext result;
		// auto cAdd = cc->Rescale(c1);
		cc->Decrypt(keys.secretKey, c1, &result);
		std::cout << "Result " << result;

		GPUct1.rescale();

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c3);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;

		// ASSERT_EQ_CIPHERTEXT(c1, cResGPU);

		CudaCheckErrorMod;
	}

	for (int i = 0; i < GPUcc.L; ++i) {
		// Encoding as plaintexts
		lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 2, i);
		lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

		std::cout << "Input x1: " << ptxt1 << std::endl;

		// Encrypt the encoded vectors
		auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
		auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
		FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

		lbcrypto::Plaintext result;
		auto cAdd = cc->Rescale(c1);
		cc->Decrypt(keys.secretKey, cAdd, &result);
		std::cout << "Result " << result;

		GPUct1.rescale();

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c3);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;

		ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

		CudaCheckErrorMod;
	}
}
#endif

TEST_P(OpenFHEInterfaceTest, MultScalar) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);

	lbcrypto::Plaintext result;
	auto cAdd = cc->EvalMult(c1, std::pow((double)2.0, (double)-7.0));

	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Mult:\n";
	std::cout << "Result " << result;

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		GPUct1.multScalar(std::pow((double)2.0, (double)-7.0), false);

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c3);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;

		CudaCheckErrorMod;
		ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEInterfaceTest, Mult) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, 0);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	lbcrypto::Plaintext result;

	// cc->RescaleInPlace(c1);
	// cc->RescaleInPlace(c2);
	auto cAdd = cc->EvalMult(c1, c2);

	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Mult:\n";
	std::cout << "Result " << result;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());

	FIDESlib::CKKS::Context& cc_       = GPUcc;
	cc_                                = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;
	{
		FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
		FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
		kskEval.Initialize(rawKskEval);
		GPUcc.AddEvalKey(std::move(kskEval));
	}

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2_(cc_, raw2);

	for (int batch : FIDESlib::Testing::batch_configs) {

		for (bool moddown : { 0, 1 }) {
			fideslibParams.batch = batch;
			std::cout << "Batch " << batch << std::endl;
			GPUcc.batch = batch;
			cudaDeviceSynchronize();

			FIDESlib::CKKS::Ciphertext GPUct1(cc_);
			GPUct1.copy(GPUct1_);

			GPUct1.mult(GPUct2_, false, moddown);

			if (!moddown)
				GPUct1.modDown();

			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU(c3);

			CudaCheckErrorMod;
			GetOpenFHECipherText(cResGPU, raw_res1);
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

			std::cout << "Result GPU " << resultGPU;

			CudaCheckErrorMod;
			ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

			CudaCheckErrorMod;
		}
	}
}

TEST_P(OpenFHEInterfaceTest, Square) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);

	lbcrypto::Plaintext result;
	auto cAdd = cc->EvalSquare(c1);

	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Mult:\n";
	std::cout << "Result " << result;

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

	kskEval.Initialize(rawKskEval);
	cc_->AddEvalKey(std::move(kskEval));

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);
		GPUct1.square(false);
		cudaDeviceSynchronize();

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c3);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;

		CudaCheckErrorMod;
		ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEInterfaceTest, MultRescale) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);

	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct2(cc_, raw2);

	lbcrypto::Plaintext result;
	auto cAdd = cc->EvalMult(c1, c2);
	cc->RescaleInPlace(cAdd);
	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Mult:\n";
	std::cout << "Result " << result;

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

	kskEval.Initialize(rawKskEval);
	cc_->AddEvalKey(std::move(kskEval));

	GPUct1.mult(GPUct2, true);

	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c3);

	GetOpenFHECipherText(cResGPU, raw_res1);
	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	std::cout << "Result GPU " << resultGPU;

	ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

	CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, Rotate) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	// cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

	lbcrypto::Plaintext result;
	auto cAdd = cc->EvalRotate(c1, 1);

	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Rotate:\n";
	std::cout << "Result " << result;

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);

	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);

	kskEval.Initialize(rawKskEval);
	GPUcc.AddRotationKey(1, std::move(kskEval));

	GPUct1.rotate(1, true);

	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c3);

	GetOpenFHECipherText(cResGPU, raw_res1);
	lbcrypto::Plaintext resultGPU;
	//  ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
	ASSERT_ERROR_OK(result, resultGPU)

	std::cout << "Result GPU " << resultGPU;

	CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, Conjugate) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	// cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x3 = { 0.0 };

	auto FHE                    = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
	auto conjKey                = FHE->ConjugateKeyGen(keys.secretKey);
	auto& evalKeyMap            = cc->GetEvalAutomorphismKeyMap(keys.publicKey->GetKeyTag());
	evalKeyMap[GPUcc.N * 2 - 1] = conjKey;

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetConjugateKeySwitchKey(keys.publicKey);
	kskEval.Initialize(rawKskEval);
	FIDESlib::CKKS::Ciphertext GPUct2(cc_);
	GPUcc.AddRotationKey(2 * GPUcc.N - 1, std::move(kskEval));

	for (int i = 0; i <= GPUcc.L; ++i) {
		// Encoding as plaintexts
		lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);
		lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

		std::cout << "Input x1: " << ptxt1 << std::endl;

		// Encrypt the encoded vectors
		auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
		auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

		FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

		lbcrypto::Plaintext result;

		auto conj = FHE->Conjugate(c1, evalKeyMap);

		GPUct2.conjugate(GPUct1);

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct2.store(raw_res1);
		auto cResGPU(c3);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		// ASSERT_EQ_CIPHERTEXT(conj, cResGPU);

		cc->Decrypt(keys.secretKey, conj, &result);
		std::cout << "Rotate:\n";
		std::cout << "Result " << result;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;

		ASSERT_ERROR_OK(result, resultGPU);

		CudaCheckErrorMod;
	}
	CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, HoistedRotate) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	// cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4 });

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto r1 = cc->Encrypt(keys.publicKey, ptxt3);
	auto r2 = cc->Encrypt(keys.publicKey, ptxt3);
	auto r3 = cc->Encrypt(keys.publicKey, ptxt3);
	auto r4 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

	FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, r1);
	FIDESlib::CKKS::Ciphertext GPUr1(cc_, raw2);
	FIDESlib::CKKS::Ciphertext GPUr2(cc_, raw2);
	FIDESlib::CKKS::Ciphertext GPUr3(cc_, raw2);
	FIDESlib::CKKS::Ciphertext GPUr4(cc_, raw2);

	lbcrypto::Plaintext result;
	auto cpu_r1 = cc->EvalRotate(c1, 1);
	auto cpu_r2 = cc->EvalRotate(c1, 2);
	auto cpu_r3 = cc->EvalRotate(c1, 3);
	auto cpu_r4 = cc->EvalRotate(c1, 4);

	std::cout << "Rotate:\n";
	cc->Decrypt(keys.secretKey, cpu_r1, &result);
	std::cout << "Result " << result;
	cc->Decrypt(keys.secretKey, cpu_r2, &result);
	std::cout << "Result " << result;
	cc->Decrypt(keys.secretKey, cpu_r3, &result);
	std::cout << "Result " << result;
	cc->Decrypt(keys.secretKey, cpu_r4, &result);
	std::cout << "Result " << result;

	FIDESlib::CKKS::KeySwitchingKey kskRot1(cc_);
	FIDESlib::CKKS::KeySwitchingKey kskRot2(cc_);
	FIDESlib::CKKS::KeySwitchingKey kskRot3(cc_);
	FIDESlib::CKKS::KeySwitchingKey kskRot4(cc_);

	FIDESlib::CKKS::RawKeySwitchKey rawkskRot1 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
	FIDESlib::CKKS::RawKeySwitchKey rawkskRot2 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 2, cc);
	FIDESlib::CKKS::RawKeySwitchKey rawkskRot3 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 3, cc);
	FIDESlib::CKKS::RawKeySwitchKey rawkskRot4 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 4, cc);

	kskRot1.Initialize(rawkskRot1);
	kskRot2.Initialize(rawkskRot2);
	kskRot3.Initialize(rawkskRot3);
	kskRot4.Initialize(rawkskRot4);

	GPUcc.AddRotationKey(1, std::move(kskRot1));
	GPUcc.AddRotationKey(2, std::move(kskRot2));
	GPUcc.AddRotationKey(3, std::move(kskRot3));
	GPUcc.AddRotationKey(4, std::move(kskRot4));

	for (int i = 0; i < 2; ++i) {
		CKKS::hoistRotateFused = i;
		GPUct1.rotate_hoisted({ 1, 2, 3, 4 }, { &GPUr1, &GPUr2, &GPUr3, &GPUr4 }, false);
	}
	/*
	for (int i = 1; i < 2; ++i) {
		CKKS::hoistRotateFused = i;
		GPUct1.rotate_hoisted({1}, {&GPUr1}, false);
	}
	*/

	// GPUct1.rotate_hoisted({&kskRot1}, {1}, {&GPUr1});
	// GPUct1.rotate(2, kskRot2);

	FIDESlib::CKKS::RawCipherText raw_res1;

	auto cResGPU(c1);
	lbcrypto::Plaintext resultGPU;

	GPUr1.store(raw_res1);
	GetOpenFHECipherText(cResGPU, raw_res1);
	// ASSERT_EQ_CIPHERTEXT(cpu_r1, cResGPU);
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
	std::cout << "Result GPU " << resultGPU;
	cc->Decrypt(keys.secretKey, cpu_r1, &result);
	ASSERT_ERROR_OK(resultGPU, result)

	GPUr2.store(raw_res1);
	GetOpenFHECipherText(cResGPU, raw_res1);
	// ASSERT_EQ_CIPHERTEXT(cpu_r2, cResGPU);
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
	std::cout << "Result GPU " << resultGPU;
	cc->Decrypt(keys.secretKey, cpu_r2, &result);
	ASSERT_ERROR_OK(resultGPU, result)

	GPUr3.store(raw_res1);
	GetOpenFHECipherText(cResGPU, raw_res1);
	// ASSERT_EQ_CIPHERTEXT(cpu_r3, cResGPU);
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
	std::cout << "Result GPU " << resultGPU;
	cc->Decrypt(keys.secretKey, cpu_r3, &result);
	ASSERT_ERROR_OK(resultGPU, result)

	GPUr4.store(raw_res1);
	GetOpenFHECipherText(cResGPU, raw_res1);
	// ASSERT_EQ_CIPHERTEXT(cpu_r4, cResGPU);
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
	std::cout << "Result GPU " << resultGPU;
	cc->Decrypt(keys.secretKey, cpu_r4, &result);
	ASSERT_ERROR_OK(resultGPU, result)

	CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, ExtractContextShowPtMultAllLevels) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);

	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 4.0, 2, 1.33333333, 1.0, 0.5, 0.333333333, 0.25, 0.2 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	std::cout << "Input x2: " << ptxt2 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3 = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::RawPlainText raw1_ = FIDESlib::CKKS::GetRawPlainText(cc, ptxt1);
	FIDESlib::CKKS::RawPlainText raw2  = FIDESlib::CKKS::GetRawPlainText(cc, ptxt2);

	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
	FIDESlib::CKKS::Plaintext GPUpt1_(cc_, raw1_);
	FIDESlib::CKKS::Plaintext GPUpt2(cc_, raw2);

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		auto cMult = c1->Clone();
		for (int i = GPUct1.getLevel() + 1 - GPUct1.NoiseLevel; i >= 1; --i) {
			// GPU ptMult
			GPUct1.multPt(i % 2 ? GPUpt2 : GPUpt1_, false);
			cMult = cc->EvalMult(cMult, i % 2 ? ptxt2 : ptxt1);
			if (GPUcc.rescaleTechnique == CKKS::FIXEDMANUAL) {
				GPUct1.rescale();
				cMult = cc->Rescale(cMult);
			}
			{
				FIDESlib::CKKS::RawCipherText raw_res1;
				GPUct1.store(raw_res1);
				auto cResGPU(c3->Clone());
				GetOpenFHECipherText(cResGPU, raw_res1);
				lbcrypto::Plaintext resultGPU;

				lbcrypto::Plaintext result;
				cc->Decrypt(keys.secretKey, cMult, &result);

				std::cout << "MultPt:\n";
				std::cout << "Result " << result;
				ASSERT_EQ_CIPHERTEXT(cResGPU, cMult);
				cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

				std::cout << "Result GPU " << resultGPU;
			}
		}
	}
}

TEST_P(OpenFHEInterfaceTest, MultAllLevels) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 4.0, 2.0, 1.3333333333, 1.0, 0.5, 0.3333333333, 0.25, 0.2 };
	std::vector<double> x3 = { 0.0 };

	// Encoding as plaintexts
	// Encrypt the encoded vectors

	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2);
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	auto c1   = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2_1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2_2 = cc->Encrypt(keys.publicKey, ptxt2);
	auto c3   = cc->Encrypt(keys.publicKey, ptxt3);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);
	FIDESlib::CKKS::Ciphertext GPUct3(cc_, raw1);

	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	kskEval.Initialize(rawKskEval);
	cc_->AddEvalKey(std::move(kskEval));

	for (int i = 0; i < GPUcc.L - (GPUcc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT); ++i) {
		ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);

		auto c2 = i % 2 == 0 ? c2_2 : c2_1;

		FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, c2);
		FIDESlib::CKKS::Ciphertext GPUct2(cc_, raw2);
		lbcrypto::Plaintext result;

		c1 = cc->EvalMult(c1, c2);
		cc->RescaleInPlace(c1);
		cc->Decrypt(keys.secretKey, c1, &result);

		std::cout << "Mult " << i << " levels used:\n";
		std::cout << "Result " << result;

		auto cResGPU(c3);

		GPUct1.mult(GPUct2, GPUcc.rescaleTechnique == CKKS::FIXEDMANUAL);

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);

		GetOpenFHECipherText(cResGPU, raw_res1);

		lbcrypto::Plaintext resultGPU;

		// try {
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		if (resultGPU->GetLogError() != resultGPU->GetLogError())
			OPENFHE_THROW("nan after decryption");
		std::cout << "Result GPU " << resultGPU;
		ASSERT_EQ_CIPHERTEXT(c1, cResGPU);
		//} catch (lbcrypto::OpenFHEException& e) {
		//    std::cout << "OpenFHE exception, continuing for debugging" << std::endl;
		//}

		GPUct3.mult(GPUct2, GPUcc.rescaleTechnique == CKKS::FIXEDMANUAL);

		FIDESlib::CKKS::RawCipherText raw_res2;
		GPUct3.store(raw_res2);

		GetOpenFHECipherText(cResGPU, raw_res2);

		lbcrypto::Plaintext resultGPU2;

		// try {
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU2);
		if (resultGPU->GetLogError() != resultGPU->GetLogError())
			OPENFHE_THROW("nan after decryption");
		std::cout << "Result GPU " << resultGPU2;
		ASSERT_EQ_CIPHERTEXT(c1, cResGPU);
		//}
		// catch (lbcrypto::OpenFHEException& e) {
		//    std::cout << "OpenFHE exception, continuing for debugging" << std::endl;
		//}

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEInterfaceTest, RotateAllLevels) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	// cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1 });

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	std::vector<double> x3 = { 0.0 };

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
	kskEval.Initialize(rawKskEval);
	GPUcc.AddRotationKey(1, std::move(kskEval));

	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	for (int i = 0; i <= GPUcc.L; ++i) {
		lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);
		// Encoding as plaintexts
		std::cout << "Input x1: " << ptxt1 << std::endl;

		// Encrypt the encoded vectors
		auto c1                            = cc->Encrypt(keys.publicKey, ptxt1);
		auto c3                            = cc->Encrypt(keys.publicKey, ptxt3);
		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
		FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

		lbcrypto::Plaintext result;
		auto cAdd = cc->EvalRotate(c1, 1);

		cc->Decrypt(keys.secretKey, cAdd, &result);

		std::cout << "Rotate " << i << " levels down:\n";
		std::cout << "Result " << result;

		GPUct1.rotate(1, true);
		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c3);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		CudaCheckErrorMod;
		// ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		ASSERT_ERROR_OK(result, resultGPU);

		std::cout << "Result GPU " << resultGPU;
	}
}

TEST_P(OpenFHEInterfaceTest, SquareAllLevels) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	fideslibParams.batch = 3;
	std::cout << "Batch " << 3 << std::endl;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

	kskEval.Initialize(rawKskEval);
	cc_->AddEvalKey(std::move(kskEval));

	for (int i = 0; i < GPUcc.L; ++i) {
		std::cout << "Dropped " << i << " levels.\n";
		// Encoding as plaintexts
		lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);

		std::cout << "Input x1: " << ptxt1 << std::endl;
		// Encrypt the encoded vectors
		auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
		auto c3 = cc->Encrypt(keys.publicKey, ptxt1);

		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

		FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

		lbcrypto::Plaintext result;
		auto cAdd = cc->EvalSquare(c1);

		cc->Decrypt(keys.secretKey, cAdd, &result);

		std::cout << "Mult:\n";
		std::cout << "Result " << result;

		GPUct1.square(false);

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c3);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;

		CudaCheckErrorMod;
		ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEInterfaceTest, HoistedRotateAllLevels) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	// cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4 });

	fideslibParams.batch                = 3;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	std::vector<double> x3    = { 0.0 };
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	FIDESlib::CKKS::KeySwitchingKey kskRot1(cc_);
	FIDESlib::CKKS::KeySwitchingKey kskRot2(cc_);
	FIDESlib::CKKS::KeySwitchingKey kskRot3(cc_);
	FIDESlib::CKKS::KeySwitchingKey kskRot4(cc_);

	FIDESlib::CKKS::RawKeySwitchKey rawkskRot1 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
	FIDESlib::CKKS::RawKeySwitchKey rawkskRot2 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 2, cc);
	FIDESlib::CKKS::RawKeySwitchKey rawkskRot3 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 3, cc);
	FIDESlib::CKKS::RawKeySwitchKey rawkskRot4 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 4, cc);

	kskRot1.Initialize(rawkskRot1);
	kskRot2.Initialize(rawkskRot2);
	kskRot3.Initialize(rawkskRot3);
	kskRot4.Initialize(rawkskRot4);

	GPUcc.AddRotationKey(1, std::move(kskRot1));
	GPUcc.AddRotationKey(2, std::move(kskRot2));
	GPUcc.AddRotationKey(3, std::move(kskRot3));
	GPUcc.AddRotationKey(4, std::move(kskRot4));

	for (int i = 0; i <= GPUcc.L; ++i) {
		std::cout << "Dropped levels: " << i << std::endl;
		// Encoding as plaintexts
		lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);

		std::cout << "Input x1: " << ptxt1 << std::endl;

		// Encrypt the encoded vectors
		auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
		auto r1 = cc->Encrypt(keys.publicKey, ptxt1);
		auto r2 = cc->Encrypt(keys.publicKey, ptxt1);
		auto r3 = cc->Encrypt(keys.publicKey, ptxt1);
		auto r4 = cc->Encrypt(keys.publicKey, ptxt1);

		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
		FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

		FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, r1);
		FIDESlib::CKKS::Ciphertext GPUr1(cc_, raw2);
		FIDESlib::CKKS::Ciphertext GPUr2(cc_, raw2);
		FIDESlib::CKKS::Ciphertext GPUr3(cc_, raw2);
		FIDESlib::CKKS::Ciphertext GPUr4(cc_, raw2);

		lbcrypto::Plaintext result1, result2, result3, result4;
		auto cpu_r1 = cc->EvalRotate(c1, 1);
		auto cpu_r2 = cc->EvalRotate(c1, 2);
		auto cpu_r3 = cc->EvalRotate(c1, 3);
		auto cpu_r4 = cc->EvalRotate(c1, 4);

		std::cout << "Rotate:\n";
		cc->Decrypt(keys.secretKey, cpu_r1, &result1);
		std::cout << "Result " << result1;
		cc->Decrypt(keys.secretKey, cpu_r2, &result2);
		std::cout << "Result " << result2;
		cc->Decrypt(keys.secretKey, cpu_r3, &result3);
		std::cout << "Result " << result3;
		cc->Decrypt(keys.secretKey, cpu_r4, &result4);
		std::cout << "Result " << result4;

		GPUct1.rotate_hoisted({ 1, 2, 3, 4 }, { &GPUr1, &GPUr2, &GPUr3, &GPUr4 }, false);
		// GPUct1.rotate_hoisted({&kskRot1}, {1}, {&GPUr1});
		// GPUct1.rotate(2, kskRot2);

		FIDESlib::CKKS::RawCipherText raw_res1;

		auto cResGPU(c1);
		lbcrypto::Plaintext resultGPU;

		GPUr1.store(raw_res1);
		GetOpenFHECipherText(cResGPU, raw_res1);
		// ASSERT_EQ_CIPHERTEXT(cpu_r1, cResGPU);

		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;
		ASSERT_ERROR_OK(result1, resultGPU);

		GPUr2.store(raw_res1);
		GetOpenFHECipherText(cResGPU, raw_res1);
		// ASSERT_EQ_CIPHERTEXT(cpu_r2, cResGPU);
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;
		ASSERT_ERROR_OK(result2, resultGPU);

		GPUr3.store(raw_res1);
		GetOpenFHECipherText(cResGPU, raw_res1);
		// ASSERT_EQ_CIPHERTEXT(cpu_r3, cResGPU);
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;
		ASSERT_ERROR_OK(result3, resultGPU);

		GPUr4.store(raw_res1);
		GetOpenFHECipherText(cResGPU, raw_res1);
		// ASSERT_EQ_CIPHERTEXT(cpu_r4, cResGPU);
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;
		ASSERT_ERROR_OK(result4, resultGPU);

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEInterfaceTest, AccumAllLevels) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	// cc->EvalMultKeyGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, { 1, 2, 3, 4 });

	fideslibParams.batch                = 3;
	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	std::vector<double> x3    = { 0.0 };
	lbcrypto::Plaintext ptxt3 = cc->MakeCKKSPackedPlaintext(x3);

	auto rotations = CKKS::GetAccumulateRotationIndices(4, 1, 8);

	CKKS::GenAndAddRotationKeys(cc, keys, cc_, rotations);

	for (int i = 0; i <= GPUcc.L; ++i) {
		std::cout << "Dropped levels: " << i << std::endl;
		// Encoding as plaintexts
		lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, i);

		std::cout << "Input x1: " << ptxt1 << std::endl;

		// Encrypt the encoded vectors
		auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
		FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

		lbcrypto::Plaintext result1;

		auto cpu_tmp = cc->EvalRotate(c1, 1);
		auto cpu_r1  = cc->EvalAdd(c1, cpu_tmp);
		for (int j = 2; j < 8; ++j) {
			cpu_tmp = cc->EvalRotate(cpu_tmp, 1);
			cpu_r1  = cc->EvalAdd(cpu_tmp, cpu_r1);
		}

		std::cout << "Rotate:\n";
		cc->Decrypt(keys.secretKey, cpu_r1, &result1);
		std::cout << "Result " << result1;

		CKKS::Accumulate(GPUct1, 4, 1, 8);

		// GPUct1.rotate_hoisted({&kskRot1}, {1}, {&GPUr1});
		// GPUct1.rotate(2, kskRot2);

		FIDESlib::CKKS::RawCipherText raw_res1;

		auto cResGPU(c1);

		lbcrypto::Plaintext resultGPU;

		GPUct1.store(raw_res1);
		GetOpenFHECipherText(cResGPU, raw_res1);
		cResGPU->SetSlots(8);
		// ASSERT_EQ_CIPHERTEXT(cpu_r1, cResGPU);

		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;
		ASSERT_ERROR_OK(result1, resultGPU);

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEInterfaceTest, MatVec) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	// cc->Enable(KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	cc->Enable(lbcrypto::KEYSWITCH);
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);

	kskEval.Initialize(rawKskEval);
	cc_->AddEvalKey(std::move(kskEval));

	cc->EvalRotateKeyGen(keys.secretKey, { 1 });
	///// PROBAR /////
	std::vector<double> x1 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

	std::vector<double> x[8] = { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 } };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 3);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	// std::cout << "Input x2: " << ptxt2 << std::endl;
	//  Encrypt the encoded vectors
	std::vector<lbcrypto::Plaintext> ptxt;
	for (int i = 0; i < 8; ++i) {
		ptxt.emplace_back(cc->MakeCKKSPackedPlaintext(x[i], 1, GPUcc.L - 3));
	}
	std::cout << "Input x2: " << ptxt[0] << std::endl;

	FIDESlib::CKKS::KeySwitchingKey kskRot1(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawkskRot1 = FIDESlib::CKKS::GetRotationKeySwitchKey(keys, 1, cc);
	kskRot1.Initialize(rawkskRot1);
	GPUcc.AddRotationKey(1, std::move(kskRot1));

	if (1) {
		using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
		std::vector<Cipher> ct;
		std::vector<Cipher> ct2;
		for (int i = 0; i < 8; ++i) {
			ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));
			ct2.emplace_back(cc->Encrypt(keys.publicKey, ptxt[i]));
		}

		std::vector<FIDESlib::CKKS::Ciphertext> GPUct;
		GPUct.reserve(8);
		std::vector<FIDESlib::CKKS::Ciphertext*> GPUct_(8);
		for (int i = 0; i < 8; ++i) {
			FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ct[i]);
			GPUct.emplace_back(cc_, raw1);
			GPUct_[i] = &GPUct[i];
			if (GPUct[i].NoiseLevel == 2)
				GPUct[i].rescale();
		}

		std::vector<FIDESlib::CKKS::Ciphertext> GPUct2;
		GPUct2.reserve(8);
		std::vector<FIDESlib::CKKS::Ciphertext*> GPUct2_(8);
		for (int i = 0; i < 8; ++i) {
			FIDESlib::CKKS::RawCipherText raw2 = FIDESlib::CKKS::GetRawCipherText(cc, ct2[i]);
			GPUct2.emplace_back(cc_, raw2);
			GPUct2_[i] = &GPUct2[i];
			if (GPUct2[i].NoiseLevel == 2)
				GPUct2[i].rescale();
		}

		//////////////////////////////////////////////
		cc->GetScheme()->EvalMultInPlace(ct[0], ct2[0], cc->GetEvalMultKeyVector(keys.secretKey->GetKeyTag()).front());
		for (int i = 1; i < 8; ++i) {
			// cc->EvalMultInPlace(ct[i], ptxt[i]);
			cc->GetScheme()->EvalMultInPlace(ct[i], ct2[i], cc->GetEvalMultKeyVector(keys.secretKey->GetKeyTag()).front());
			cc->EvalAddInPlace(ct[0], ct[i]);
		}
		// cc->RescaleInPlace(ct[0]);

		{
			lbcrypto::Plaintext resultCPU;
			cc->Decrypt(keys.secretKey, ct[0], &resultCPU);
			std::cout << "Result CPU " << resultCPU;
		}
		CKKS::Ciphertext result(cc_);
		result.dotProduct(GPUct_, GPUct2_, false);

		// result.rescale();
		//  GPUct[0].rescale();
		{
			FIDESlib::CKKS::RawCipherText raw1;
			result.store(raw1);

			// cc->RescaleInPlace(ct[1]); // The reference cyphertext has to be rescaled from or it bugs out.
			Cipher cResGPU(ct[1] /*cc->Encrypt(keys.publicKey, ptxt1)*/);
			FIDESlib::CKKS::GetOpenFHECipherText(cResGPU, raw1);
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU " << resultGPU;
			lbcrypto::Plaintext resultCPU;
			cc->Decrypt(keys.secretKey, ct[0], &resultCPU);
			std::cout << "Result CPU " << resultCPU;
			ASSERT_ERROR_OK(resultCPU, resultGPU);
			// ASSERT_EQ_CIPHERTEXT(cResGPU, ct[0]);
		}
	}

	CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, MatVecPt) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	// cc->Enable(KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	///// PROBAR /////
	std::vector<double> x1 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

	std::vector<double> x[8] = { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 } };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	// std::cout << "Input x2: " << ptxt2 << std::endl;
	//  Encrypt the encoded vectors
	std::vector<lbcrypto::Plaintext> ptxt;
	for (int i = 0; i < 8; ++i) {
		ptxt.emplace_back(cc->MakeCKKSPackedPlaintext(x[i]));
	}
	std::cout << "Input x2: " << ptxt[0] << std::endl;
	using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
	std::vector<Cipher> ct;
	for (int i = 0; i < 8; ++i)
		ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));

	if (1) {
		std::vector<FIDESlib::CKKS::Ciphertext> GPUct;
		//     GPUct.reserve(8);
		for (int i = 0; i < 8; ++i) {
			FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ct[i]);
			GPUct.emplace_back(cc_, raw1);
		}

		std::vector<FIDESlib::CKKS::Plaintext> GPUpt;
		for (int i = 0; i < 8; ++i) {
			FIDESlib::CKKS::RawPlainText raw2 = FIDESlib::CKKS::GetRawPlainText(cc, ptxt[i]);
			GPUpt.emplace_back(cc_, raw2);
		}

		//////////////////////////////////////////////
		cc->GetScheme()->EvalMultInPlace(ct[0], ptxt[0]);
		for (int i = 1; i < 8; ++i) {
			// cc->EvalMultInPlace(ct[i], ptxt[i]);
			cc->GetScheme()->EvalMultInPlace(ct[i], ptxt[i]);
			cc->EvalAddInPlace(ct[0], ct[i]);
		}
		// cc->RescaleInPlace(ct[0]);

		lbcrypto::Plaintext resultCPU;
		cc->Decrypt(keys.secretKey, ct[0], &resultCPU);
		std::cout << "Result CPU " << resultCPU;
		GPUct[0].multPt(GPUpt[0], false);
		// GPUct[0].rescale();

		for (int i = 1; i < 8; ++i) {
			GPUct[i].multPt(GPUpt[i], false);
			// GPUct[i].rescale();
			GPUct[0].add(GPUct[i]);
		}
		// GPUct[0].rescale();
		{
			FIDESlib::CKKS::RawCipherText raw1;
			GPUct[0].store(raw1);

			// cc->RescaleInPlace(ct[1]); // The reference cyphertext has to be rescaled from or it bugs out.
			Cipher cResGPU(ct[1] /*cc->Encrypt(keys.publicKey, ptxt1)*/);
			FIDESlib::CKKS::GetOpenFHECipherText(cResGPU, raw1);
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU " << resultGPU;

			ASSERT_EQ_CIPHERTEXT(cResGPU, ct[0]);
		}
	}
	CudaCheckErrorMod;
}

TEST_P(OpenFHEInterfaceTest, MatVecPtScalar) {
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	// cc->Enable(KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	///// PROBAR /////
	std::vector<double> x1 = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

	std::vector<double> x[8] = { { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }, { 6.0 }, { 7.0 }, { 8.0 } };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	// std::cout << "Input x2: " << ptxt2 << std::endl;

	using Cipher = lbcrypto::Ciphertext<lbcrypto::DCRTPolyImpl<bigintdyn::mubintvec<bigintdyn::ubint<unsigned long>>>>;
	std::vector<Cipher> ct;
	for (int i = 0; i < 8; ++i)
		ct.emplace_back(cc->Encrypt(keys.publicKey, ptxt1));

	if (1) {
		std::vector<FIDESlib::CKKS::Ciphertext> GPUct;
		for (int i = 0; i < 8; ++i) {
			FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ct[i]);
			GPUct.emplace_back(cc_, raw1);
		}
		/*
		std::vector<FIDESlib::CKKS::Plaintext> GPUpt;
		for (int i = 0; i < 8; ++i) {
			FIDESlib::CKKS::RawPlainText raw2 = FIDESlib::CKKS::GetRawPlainText(cc, ptxt[i]);
			GPUpt.emplace_back(cc_, raw2);
		}
*/
		//////////////////////////////////////////////
		cc->GetScheme()->EvalMultInPlace(ct[0], x[0][0]);
		for (int i = 1; i < 8; ++i) {
			// cc->EvalMultInPlace(ct[i], ptxt[i]);
			cc->GetScheme()->EvalMultInPlace(ct[i], x[i][0]);
			cc->EvalAddInPlace(ct[0], ct[i]);
		}
		// cc->RescaleInPlace(ct[0]);

		lbcrypto::Plaintext resultCPU;
		cc->Decrypt(keys.secretKey, ct[0], &resultCPU);
		std::cout << "Result CPU " << resultCPU;
		GPUct[0].multScalar(x[0][0], false);
		// GPUct[0].rescale();

		for (int i = 1; i < 8; ++i) {
			GPUct[i].multScalar(x[i][0], false);
			// GPUct[i].rescale();
			GPUct[0].add(GPUct[i]);
		}
		// GPUct[0].rescale();
		{
			FIDESlib::CKKS::RawCipherText raw1;
			GPUct[0].store(raw1);

			// cc->RescaleInPlace(ct[1]); // The reference cyphertext has to be rescaled from or it bugs out.
			Cipher cResGPU(ct[1] /*cc->Encrypt(keys.publicKey, ptxt1)*/);
			FIDESlib::CKKS::GetOpenFHECipherText(cResGPU, raw1);
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU " << resultGPU;

			ASSERT_EQ_CIPHERTEXT(cResGPU, ct[0]);
		}
	}
	CudaCheckErrorMod;
}

// Define the parameter sets
INSTANTIATE_TEST_SUITE_P(OpenFHEInterfaceTests, OpenFHEInterfaceTest, testing::Values(TTALL64BOOT));

class OpenFHEBootstrapTest : public GeneralParametrizedTest {
};

/*
TEST_P(OpenFHEBootstrapTest, ModRaise) {
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
				->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_ = GPUcc; cc_ = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

	const int slots = 32;
	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	lbcrypto::Plaintext result;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({1, 1}, {0, 0}, slots);

	std::cout << "Generate keys" << std::endl;

	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
	// coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562
0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163 0.139855
0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249 0.212888 0.00178101
0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932 0.0245978 0.000355532 -0.0082485
-0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05 -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06
-1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10 -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12
-7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14 std::cout << "Run bootstrap start" << std::endl; auto FHE =
std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE); auto raised = FHE->EvalBootstrapSetupOnly(c1, 1, 0);

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		std::cout << "Run ModRaise GPU" << std::endl;
		uint32_t correction = 0;  // TODO
		CKKS::ModRaise(GPUct1, slots, correction);

		FIDESlib::CKKS::EvalLinearTransform(GPUct1, slots, false);

		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);

			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU after setup" << resultGPU;
			CudaCheckErrorMod;
			ASSERT_ERROR_OK(result, resultGPU);
			ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
		}

		CudaCheckErrorMod;
	}
}
*/
TEST_P(OpenFHEBootstrapTest, ModRaiseLimbCheck) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_ = GPUcc;
	cc_ = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	const int slots = 32;
	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 1, nullptr, slots);
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1(cc_, raw1);

	if (GPUct1.NoiseLevel == 2)
		GPUct1.rescale();
	GPUct1.dropToLevel(0, true);
	std::cout << "Before ModRaise: level=" << GPUct1.getLevel() << " NoiseLevel=" << GPUct1.NoiseLevel << std::endl;

	GPUct1.c0.INTT(GPUcc.batch, true);
	std::vector<std::vector<uint64_t>> before;
	GPUct1.c0.store(before);
	GPUct1.c0.NTT(GPUcc.batch, true);
	std::cout << "Before: " << before.size() << " limbs, limb0 size=" << before[0].size() << std::endl;
	std::cout << "Limb0 first5: ";
	for (int i = 0; i < 5 && i < (int)before[0].size(); ++i) std::cout << before[0][i] << " ";
	std::cout << std::endl;

	FIDESlib::CKKS::ModRaise(GPUct1, slots, 0, true, false);
	std::cout << "After ModRaise: level=" << GPUct1.getLevel() << std::endl;

	GPUct1.c0.INTT(GPUcc.batch, true);
	std::vector<std::vector<uint64_t>> after;
	GPUct1.c0.store(after);
	GPUct1.c0.NTT(GPUcc.batch, true);
	std::cout << "After: " << after.size() << " limbs" << std::endl;

	bool limb0_ok = true;
	int limb0_mismatches = 0;
	for (size_t i = 0; i < before[0].size() && i < after[0].size(); ++i) {
		if (after[0][i] != before[0][i]) {
			if (limb0_mismatches < 10)
				std::cout << "Limb0 MISMATCH coeff " << i << ": " << after[0][i] << " vs " << before[0][i] << std::endl;
			limb0_mismatches++;
		}
	}
	if (limb0_mismatches > 0) {
		std::cout << "FAIL: limb0 has " << limb0_mismatches << " mismatches" << std::endl;
		limb0_ok = false;
	} else {
		std::cout << "PASS: limb0 bit-identical" << std::endl;
	}

	bool newlimbs_ok = true;
	for (size_t limb = 1; limb < after.size(); ++limb) {
		uint64_t q_i = GPUcc.prime[limb].p;
		int mismatches = 0;
		for (size_t j = 0; j < before[0].size() && j < after[limb].size(); ++j) {
			uint64_t q0 = GPUcc.prime[0].p;
			uint64_t a = before[0][j];
			uint64_t expected;
			if (a > q0 / 2) {
				uint64_t diff = q_i - (q0 % q_i);
				a = a + diff;
			}
			if (a >= q_i) a = a % q_i;
			expected = a;
			if (after[limb][j] != expected) {
				if (mismatches < 5)
					std::cout << "Limb" << limb << " MISMATCH coeff " << j << ": " << after[limb][j]
						<< " vs " << expected << " (" << before[0][j] << " mod " << q_i << ")" << std::endl;
				mismatches++;
			}
		}
		if (mismatches > 0) {
			std::cout << "FAIL: limb" << limb << " (q=" << q_i << ") has " << mismatches << " mismatches" << std::endl;
			newlimbs_ok = false;
		}
	}
	if (newlimbs_ok) std::cout << "PASS: all new limbs = limb0 mod q_i" << std::endl;

	ASSERT_TRUE(limb0_ok && newlimbs_ok);
}
// CtS-StC identity test: compose GPU CtS then GPU StC (no EvalMod in between).
// The composition should yield input * (1/k) where k = K_SPARSE for sparse encoding.
// Trap 1 (from 08d6720): check output[i]/input[i] ratio consistency across slots, not absolute error.
// Trap 2 (from 08d6720): encode at level 11 (EvalMod exit level for {2,2}), not fresh top level.
TEST_P(OpenFHEBootstrapTest, CtSStCIdentity) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	int slots = 1 << 5;
	std::cout << "Setup Bootstrap {2,2}, slots=" << slots << std::endl;
	cc->EvalBootstrapSetup({2, 2}, {0, 0}, slots);
	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	std::vector<double> x1(slots);
	for (int i = 0; i < slots; i++) x1[i] = 1.0 + i;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());

	int targetLevel = 11;
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, targetLevel, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input first8: ";
	for (int i = 0; i < 8; i++) std::cout << x1[i] << " ";
	std::cout << std::endl;

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	std::cout << "Encrypted level: " << c1->GetLevel() << std::endl;

	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);
		std::cout << "Before CtS: level=" << GPUct1.getLevel() << std::endl;

		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, false);
		CudaCheckErrorMod;
		std::cout << "After CtS: level=" << GPUct1.getLevel() << std::endl;

		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, true);
		CudaCheckErrorMod;
		std::cout << "After StC: level=" << GPUct1.getLevel() << std::endl;

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU = c2->Clone();
		GetOpenFHECipherText(cResGPU, raw_res1);

		if (slots < GPUcc.N / 2) {
			auto conj = cc->EvalRotate(cResGPU, slots);
			cc->EvalAddInPlace(cResGPU, conj);
		}

		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		auto outputComplex = resultGPU->GetCKKSPackedValue();

		std::vector<double> ratios;
		for (int i = 0; i < slots && i < (int)outputComplex.size(); i++) {
			double outReal = outputComplex[i].real();
			if (std::abs(x1[i]) > 1e-10) {
				ratios.push_back(outReal / x1[i]);
			}
		}

		if (!ratios.empty()) {
			double minR = *std::min_element(ratios.begin(), ratios.end());
			double maxR = *std::max_element(ratios.begin(), ratios.end());
			std::sort(ratios.begin(), ratios.end());
			double medianR = ratios[ratios.size() / 2];
			double spread  = (maxR - minR) / std::abs(medianR);

			std::cout << "Ratio stats: min=" << minR << " max=" << maxR
			          << " median=" << medianR << " spread=" << spread << std::endl;
			std::cout << "First 8 ratios: ";
			for (int i = 0; i < 8 && i < (int)ratios.size(); i++) std::cout << ratios[i] << " ";
			std::cout << std::endl;

			if (spread < 0.01) {
				std::cout << "PASS: ratio is constant (spread < 1%)" << std::endl;
			} else {
				std::cout << "FAIL: ratio varies across slots (spread = " << spread << ")" << std::endl;
			}
		}

		std::cout << "First 8 output: ";
		for (int i = 0; i < 8 && i < (int)outputComplex.size(); i++)
			std::cout << outputComplex[i].real() << " ";
		std::cout << std::endl;

		CudaCheckErrorMod;
	}
}

// CtS-StC identity test (DENSE): slots = N/2, {3,3} levelBudget — production config.
// Addresses 744a883 §3: the /32-slot sparse-branch test may not apply to production dense branch.
TEST_P(OpenFHEBootstrapTest, CtSStCIdentityDense) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	int slots = cc->GetRingDimension() / 2;
	std::cout << "Setup Bootstrap {3,3}, slots=" << slots << " (dense)" << std::endl;
	cc->EvalBootstrapSetup({3, 3}, {0, 0}, slots);
	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	std::vector<double> x1(slots);
	for (int i = 0; i < slots; i++) x1[i] = 1.0 + i;

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());

	std::cout << "raw_param.L = " << raw_param.L << std::endl;
	int targetLevel = 12;
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, targetLevel, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input first8: ";
	for (int i = 0; i < 8; i++) std::cout << x1[i] << " ";
	std::cout << std::endl;

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);
	std::cout << "Encrypted level: " << c1->GetLevel() << std::endl;

	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);
		std::cout << "Before CtS: level=" << GPUct1.getLevel() << std::endl;

		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, false);
		CudaCheckErrorMod;
		std::cout << "After CtS: level=" << GPUct1.getLevel() << std::endl;

		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, true);
		CudaCheckErrorMod;
		std::cout << "After StC: level=" << GPUct1.getLevel() << std::endl;

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU = c2->Clone();
		GetOpenFHECipherText(cResGPU, raw_res1);

		if (slots < GPUcc.N / 2) {
			auto conj = cc->EvalRotate(cResGPU, slots);
			cc->EvalAddInPlace(cResGPU, conj);
		}

		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		auto outputComplex = resultGPU->GetCKKSPackedValue();

		std::vector<double> ratios;
		int checkCount = std::min(256, slots);
		for (int i = 0; i < checkCount && i < (int)outputComplex.size(); i++) {
			double outReal = outputComplex[i].real();
			if (std::abs(x1[i]) > 1e-10) {
				ratios.push_back(outReal / x1[i]);
			}
		}

		if (!ratios.empty()) {
			double minR = *std::min_element(ratios.begin(), ratios.end());
			double maxR = *std::max_element(ratios.begin(), ratios.end());
			std::sort(ratios.begin(), ratios.end());
			double medianR = ratios[ratios.size() / 2];
			double spread  = (maxR - minR) / std::abs(medianR);

			std::cout << "Ratio stats (" << ratios.size() << " slots): min=" << minR << " max=" << maxR
			          << " median=" << medianR << " spread=" << spread << std::endl;
			std::cout << "First 8 ratios: ";
			for (int i = 0; i < 8 && i < (int)ratios.size(); i++) std::cout << ratios[i] << " ";
			std::cout << std::endl;

			if (spread < 0.01) {
				std::cout << "PASS: ratio is constant (spread < 1%)" << std::endl;
			} else {
				std::cout << "FAIL: ratio varies across slots (spread = " << spread << ")" << std::endl;
			}
		}

		std::cout << "First 8 output: ";
		for (int i = 0; i < 8 && i < (int)outputComplex.size(); i++)
			std::cout << outputComplex[i].real() << " ";
		std::cout << std::endl;

		CudaCheckErrorMod;
	}
}

// CtS error analysis: compare CPU CtS vs GPU CtS per-slot, compute ratio statistics.
// The existing CoeffsToSlots test shows 20-26x error but doesn't analyze the pattern.
// This test determines whether the error is a constant scaling factor (systematic)
// or slot-dependent (precision defect).
TEST_P(OpenFHEBootstrapTest, CtSErrorAnalysis) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	int slots = cc->GetRingDimension() / 2;
	std::cout << "Setup Bootstrap {3,3}, slots=" << slots << std::endl;
	cc->EvalBootstrapSetup({3, 3}, {0, 0}, slots);
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	std::vector<double> x1 = {0.25, 0.5, 0.75, 0.1, -0.1, -0.75, -0.5, -0.25};

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	int encLevel = GPUcc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT ? 2 : 1;
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, encLevel, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
	auto raised = c1->Clone();

	// CPU CtS
	std::cout << "Running CPU CtS..." << std::endl;
	auto ctxtEncCPU = FHE->EvalCoeffsToSlots(FHE->m_bootPrecomMap.at(slots)->m_U0hatTPreFFT, raised);
	cc->RescaleInPlace(ctxtEncCPU);
	// NOTE: no conjugate+add on CPU side, to match GPU (which skips it for dense N/2 slots)
	lbcrypto::Plaintext resultCPU;
	cc->Decrypt(keys.secretKey, ctxtEncCPU, &resultCPU);
	auto cpuValues = resultCPU->GetCKKSPackedValue();
	std::cout << "CPU CtS done. LogPrecision=" << resultCPU->GetLogPrecision() << std::endl;

	// GPU CtS
	std::cout << "Running GPU CtS..." << std::endl;
	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, raised);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, false);
		CudaCheckErrorMod;

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU = c2->Clone();
		GetOpenFHECipherText(cResGPU, raw_res1);

		if (slots < GPUcc.N / 2) {
			auto conj = cc->EvalRotate(cResGPU, slots);
			cc->EvalAddInPlace(cResGPU, conj);
		}

		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		auto gpuValues = resultGPU->GetCKKSPackedValue();

		// Compute per-slot ratio GPU/CPU for non-zero CPU values
		std::vector<double> ratios;
		std::vector<double> absErrors;
		double maxError = 0;
		int checkCount = std::min(64, slots);
		for (int i = 0; i < checkCount && i < (int)cpuValues.size() && i < (int)gpuValues.size(); i++) {
			double cpuReal = cpuValues[i].real();
			double gpuReal = gpuValues[i].real();
			double absDiff = std::abs(gpuReal - cpuReal);
			maxError = std::max(maxError, absDiff);
			absErrors.push_back(absDiff);
			if (std::abs(cpuReal) > 1e-6) {
				ratios.push_back(gpuReal / cpuReal);
			}
		}

		double expected = pow(2.0, -resultCPU->GetLogPrecision() + 1);
		std::cout << "Batch " << batch << std::endl;
		std::cout << "Max error: " << maxError << " (Expected: " << expected << "), ratio: " << maxError / expected << "x" << std::endl;

		if (!ratios.empty()) {
			double minR = *std::min_element(ratios.begin(), ratios.end());
			double maxR = *std::max_element(ratios.begin(), ratios.end());
			std::sort(ratios.begin(), ratios.end());
			double medianR = ratios[ratios.size() / 2];
			double spread = (maxR - minR) / std::abs(medianR);

			std::cout << "GPU/CPU ratio stats (" << ratios.size() << " non-zero slots):" << std::endl;
			std::cout << "  min=" << minR << " max=" << maxR << " median=" << medianR << " spread=" << spread << std::endl;
			std::cout << "  First 8 ratios: ";
			for (int i = 0; i < 8 && i < (int)ratios.size(); i++) std::cout << ratios[i] << " ";
			std::cout << std::endl;

			if (spread < 0.01) {
				std::cout << "  → CONSTANT scaling error (systematic, not precision)" << std::endl;
			} else {
				std::cout << "  → SLOT-DEPENDENT error (precision defect)" << std::endl;
			}
		}

		// Print first 8 CPU and GPU values side by side
		std::cout << "First 8 values (CPU vs GPU):" << std::endl;
		for (int i = 0; i < 8 && i < (int)cpuValues.size() && i < (int)gpuValues.size(); i++) {
			std::cout << "  slot" << i << ": CPU=" << cpuValues[i].real() << " GPU=" << gpuValues[i].real()
			          << " diff=" << std::abs(gpuValues[i].real() - cpuValues[i].real()) << std::endl;
		}

		CudaCheckErrorMod;
	}
}

// CtS conjugate fold analysis: compare BOTH halves of the production fold.
// Production (Bootstrap.cu:119-130) uses:
//   re = ctxt + conj(ctxt)  → 2·Re   (tested by existing CoeffsToSlots, 20.9x error)
//   im = ctxt - conj(ctxt)  → 2·Im   (NEVER TESTED — fed to approxModReduction)
// A defect that cancels in +conj but flips in -conj would be invisible to all tests
// but would corrupt production. This test checks both halves separately.
// Uses same input as existing CoeffsToSlots (8 values, level 1) for 36-bit reference.
TEST_P(OpenFHEBootstrapTest, CtSConjugateFold) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	int slots = cc->GetRingDimension() / 2;
	std::cout << "Setup Bootstrap {3,3}, slots=" << slots << std::endl;
	cc->EvalBootstrapSetup({3, 3}, {0, 0}, slots);
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	std::vector<double> x1 = {0.25, 0.5, 0.75, 0.1, -0.1, -0.75, -0.5, -0.25};

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	int encLevel = GPUcc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT ? 2 : 1;
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, encLevel, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
	auto raised = c1->Clone();

	// CPU CtS
	std::cout << "Running CPU CtS..." << std::endl;
	auto ctxtEncCPU = FHE->EvalCoeffsToSlots(FHE->m_bootPrecomMap.at(slots)->m_U0hatTPreFFT, raised);
	cc->RescaleInPlace(ctxtEncCPU);
	auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEncCPU->GetKeyTag());
	auto conjCPU = FHE->Conjugate(ctxtEncCPU, evalKeyMap);

	// CPU re half: ctxt + conj
	auto reCPU = ctxtEncCPU->Clone();
	cc->EvalAddInPlace(reCPU, conjCPU);
	lbcrypto::Plaintext resultReCPU;
	cc->Decrypt(keys.secretKey, reCPU, &resultReCPU);

	// CPU im half: ctxt - conj
	auto imCPU = ctxtEncCPU->Clone();
	cc->EvalSubInPlace(imCPU, conjCPU);
	lbcrypto::Plaintext resultImCPU;
	cc->Decrypt(keys.secretKey, imCPU, &resultImCPU);

	std::cout << "CPU CtS done. Re LogPrecision=" << resultReCPU->GetLogPrecision()
	          << " Im LogPrecision=" << resultImCPU->GetLogPrecision() << std::endl;

	// GPU CtS
	std::cout << "Running GPU CtS..." << std::endl;
	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, raised);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, false);
		CudaCheckErrorMod;

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU = c2->Clone();
		GetOpenFHECipherText(cResGPU, raw_res1);

		auto evalKeyMapGPU = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
		auto conjGPU = FHE->Conjugate(cResGPU, evalKeyMapGPU);

		// GPU re half: ctxt + conj
		auto reGPU = cResGPU->Clone();
		cc->EvalAddInPlace(reGPU, conjGPU);
		lbcrypto::Plaintext resultReGPU;
		cc->Decrypt(keys.secretKey, reGPU, &resultReGPU);

		// GPU im half: ctxt - conj
		auto imGPU = cResGPU->Clone();
		cc->EvalSubInPlace(imGPU, conjGPU);
		lbcrypto::Plaintext resultImGPU;
		cc->Decrypt(keys.secretKey, imGPU, &resultImGPU);

		// Compare both halves
		std::cout << "Batch " << batch << std::endl;

		// Re half (existing test checks this)
		{
			double acc = 0.0, Max = 0.0;
			for (size_t i = 0; i < resultReCPU->GetSlots(); ++i) {
				double diff = std::abs(resultReGPU->GetRealPackedValue().at(i) - resultReCPU->GetRealPackedValue().at(i));
				acc += diff * diff;
				Max = std::max(Max, diff);
			}
			acc = std::sqrt(acc / resultReCPU->GetSlots());
			double expected = pow(2.0, -resultReCPU->GetLogPrecision() + 1);
			std::cout << "Re half: Max error: " << Max << " (Expected: " << expected
			          << "), ratio: " << Max / expected << "x, dev: " << acc << std::endl;
		}

		// Im half (NEVER TESTED BEFORE)
		{
			double acc = 0.0, Max = 0.0;
			for (size_t i = 0; i < resultImCPU->GetSlots(); ++i) {
				double diff = std::abs(resultImGPU->GetRealPackedValue().at(i) - resultImCPU->GetRealPackedValue().at(i));
				acc += diff * diff;
				Max = std::max(Max, diff);
			}
			acc = std::sqrt(acc / resultImCPU->GetSlots());
			double expected = pow(2.0, -resultImCPU->GetLogPrecision() + 1);
			std::cout << "Im half: Max error: " << Max << " (Expected: " << expected
			          << "), ratio: " << Max / expected << "x, dev: " << acc << std::endl;
		}

		// Print first 8 values for both halves
		std::cout << "First 8 Re (CPU vs GPU):" << std::endl;
		for (int i = 0; i < 8; i++) {
			std::cout << "  slot" << i << ": CPU=" << resultReCPU->GetRealPackedValue().at(i)
			          << " GPU=" << resultReGPU->GetRealPackedValue().at(i) << std::endl;
		}
		std::cout << "First 8 Im (CPU vs GPU):" << std::endl;
		for (int i = 0; i < 8; i++) {
			std::cout << "  slot" << i << ": CPU=" << resultImCPU->GetRealPackedValue().at(i)
			          << " GPU=" << resultImGPU->GetRealPackedValue().at(i) << std::endl;
		}

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEBootstrapTest, ApproxModEval) {

	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, -0.75, -0.50, -0.25, 0.1, -0.1 };

	// Encoding as plaintexts
	int slots                 = cc->GetRingDimension() / 2;
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;
	lbcrypto::Plaintext result;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({ 5, 5 }, { 0, 0 }, slots, 0, true);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);
	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc.AddEvalKey(std::move(kskEval));
	// Encrypt the encoded vectors
	auto ctxtEnc  = cc->Encrypt(keys.publicKey, ptxt1);
	auto ctxtEncI = cc->Encrypt(keys.publicKey, ptxt2);
	auto c2       = cc->Encrypt(keys.publicKey, ptxt2);

	// coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562
	// 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163
	// 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249
	// 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932
	// 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05
	// -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10
	// -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
	std::cout << "Dense ModEval" << std::endl;

	/////////////////////////////////////
	constexpr bool previous = false;
	FIDESlib::CKKS::RawCipherText raw1;
	FIDESlib::CKKS::RawCipherText raw2;
	FIDESlib::CKKS::Ciphertext GPUct1(cc_);
	FIDESlib::CKKS::Ciphertext GPUct2(cc_);
	if constexpr (previous) {
		raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEnc);
		GPUct1.load(raw1);
		FIDESlib::CKKS::Ciphertext aux(cc_);
		cudaDeviceSynchronize();

		aux.conjugate(GPUct1);
		cudaDeviceSynchronize();
		cudaDeviceSynchronize();
		GPUct2.sub(GPUct1, aux);
		cudaDeviceSynchronize();
		GPUct1.add(aux);
		cudaDeviceSynchronize();

		GPUct2.multMonomial(3 * 2 * GPUcc.N / 4);
		cudaDeviceSynchronize();
		// ctxt.copy(ctxtEncI);
		cudaDeviceSynchronize();
		if (GPUcc.rescaleTechnique == CKKS::FIXEDMANUAL) {
			GPUct1.rescale();
			GPUct2.rescale();
		}
	}

	if (0) {
		auto FHE        = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
		auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
		auto conj       = FHE->Conjugate(ctxtEnc, evalKeyMap);
		// auto ctxtEncI = ctxtEnc;
		auto ctxtEncI = cc->EvalSub(ctxtEnc, conj);
		cc->EvalAddInPlace(ctxtEnc, conj);

		cc->GetScheme()->MultByMonomialInPlace(ctxtEncI, 3 * GPUcc.N * 2 / 4);

		if (ctxtEnc->GetNoiseScaleDeg() > 1) {
			cc->ModReduceInPlace(ctxtEnc);
			cc->ModReduceInPlace(ctxtEncI);
		}
	} else {
	}
	//////////////////////////
	if constexpr (!previous) {
		raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEnc);
		GPUct1.load(raw1);
		raw2 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEncI);
		GPUct2.load(raw2);
	}

	{
		lbcrypto::Plaintext result;
		cc->Decrypt(keys.secretKey, ctxtEnc, &result);
		lbcrypto::Plaintext result2;
		cc->Decrypt(keys.secretKey, ctxtEncI, &result2);

		std::cout << "Starting point after coeffs to slots:\n";
		// std::cout << "Result " << result;
		// std::cout << "Result2 " << result2;
		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			// std::cout << "Result GPU before cheby" << resultGPU;
			CudaCheckErrorMod;
			ASSERT_ERROR_OK(result, resultGPU);
			//  ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
		}
		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct2.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			// std::cout << "Result GPU before cheby" << resultGPU;
			CudaCheckErrorMod;
			ASSERT_ERROR_OK(result2, resultGPU);
			//  ASSERT_EQ_CIPHERTEXT(ctxtEncI, cResGPU);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////
	// Evaluate Chebyshev series for the sine wave
	ctxtEnc  = cc->EvalChebyshevSeries(ctxtEnc, GPUcc.GetCoeffsChebyshev(), -1.0, 1.0);
	ctxtEncI = cc->EvalChebyshevSeries(ctxtEncI, GPUcc.GetCoeffsChebyshev(), -1.0, 1.0);

	/*
		for (const auto& i : ctxtEnc->GetElements().at(0).m_vectors) {
			std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
		}
		std::cout << std::endl;

		for (const auto& i : ctxtEncI->GetElements().at(0).m_vectors) {
			std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
		}
		std::cout << std::endl;
		*/
	// Double-angle iterations
	if (true
		//(cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY) || (cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY)
	) {
		if (false) {
			// cryptoParams->GetScalingTechnique() != FIXEDMANUAL) {
			cc->GetScheme()->ModReduceInternalInPlace(ctxtEnc, lbcrypto::BASE_NUM_LEVELS_TO_DROP);
			cc->GetScheme()->ModReduceInternalInPlace(ctxtEncI, lbcrypto::BASE_NUM_LEVELS_TO_DROP);
		}
		uint32_t numIter;
		/*
			  if (cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY)
				numIter = R_UNIFORM;
			else
				numIter = R_SPARSE;
				*/
		numIter = GPUcc.GetDoubleAngleIts();
		std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->ApplyDoubleAngleIterations(ctxtEnc, numIter);
		// lbcrypto::FHECKKSRNS::ApplyDoubleAngleIterations(ctxtEnc, numIter);
		std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->ApplyDoubleAngleIterations(ctxtEncI, numIter);

		for (const auto& i : ctxtEnc->GetElements().at(0).m_vectors) {
			std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
		}
		std::cout << std::endl;

		for (const auto& i : ctxtEncI->GetElements().at(0).m_vectors) {
			std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
		}
		std::cout << std::endl;
	}

	cc->GetScheme()->MultByMonomialInPlace(ctxtEncI, cc->GetRingDimension() / 2);
	cc->EvalAddInPlace(ctxtEnc, ctxtEncI);

	// scale the message back up after Chebyshev interpolation
	cc->GetScheme()->MultByIntegerInPlace(ctxtEnc, 1.0);

	for (const auto& i : ctxtEnc->GetElements().at(0).m_vectors) {
		std::cout << "(" << i.m_params->GetModulus() << ", " << i.m_values->at(0) << ") ";
	}
	std::cout << std::endl;

	////////////////////////////////////////////////////////////////////

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct1_(cc_);
		FIDESlib::CKKS::Ciphertext GPUct2_(cc_);
		GPUct1_.copy(GPUct1);
		GPUct2_.copy(GPUct2);
		cudaDeviceSynchronize();

		FIDESlib::CKKS::approxModReduction(GPUct1_, GPUct2_, kskEval, 1.0);

		{
			lbcrypto::Plaintext result;
			// std::cout << "Result GPU after cheby" << resultGPU->GetStringValue().substr(0, 120);
			cc->Decrypt(keys.secretKey, ctxtEnc, &result);

			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1_.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);
			lbcrypto::Plaintext resultGPU;
			CudaCheckErrorMod;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

			// std::cout << "Result GPU after cheby" << resultGPU->GetStringValue().substr(0, 120);
			CudaCheckErrorMod;
			ASSERT_ERROR_OK(result, resultGPU);
			// ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
		}

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEBootstrapTest, ApproxModEvalSparse) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	///// PROBAR /////
	std::vector<double> x1 = { 0.25 / 2.0, 0.5 / 2.0, 0.75 / 2.0, -0.75 / 2.0, -0.50 / 2.0, -0.25 / 2.0, 0.1 / 2.0, -0.1 / 2.0 };

	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, 8);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, 8);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	lbcrypto::Plaintext result;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({ 1, 1 }, { 0, 0 }, 8);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, 8);

	// coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562
	// 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163
	// 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249
	// 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932
	// 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05
	// -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10
	// -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
	// FIDESlib::CKKS::Bootstrap(GPUct1, 8);

	auto raised = c1; // std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->EvalBootstrapSetupOnly(c1, 1, 0);

	// cc->GetScheme()->ModReduceInternalInPlace(c1, 1);

	// const std::shared_ptr<lbcrypto::CKKSBootstrapPrecom> precom =
	//   std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->m_bootPrecomMap.find(8)->second;
	// bool isLTBootstrap = (precom->m_paramsEnc.lvlb /*[lbcrypto::CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1) &&
	//   (precom->m_paramsDec.lvlb /*[lbcrypto::CKKS_BOOT_PARAMS::LEVEL_BUDGET]*/ == 1);
	// auto ctxtEnc = (isLTBootstrap) ? std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->EvalLinearTransform(precom->m_U0hatTPre,
	// raised) : std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->EvalCoeffsToSlots(precom->m_U0hatTPreFFT, raised);

	auto ctxtEnc    = c1->Clone();
	auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
	auto conj       = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->Conjugate(ctxtEnc, evalKeyMap);
	cc->EvalAddInPlace(ctxtEnc, conj);

	const auto cryptoParams = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
	if (cryptoParams->GetScalingTechnique() == lbcrypto::FIXEDMANUAL) {
		while (ctxtEnc->GetNoiseScaleDeg() > 1) {
			cc->ModReduceInPlace(ctxtEnc);
		}
	} else {
		if (ctxtEnc->GetNoiseScaleDeg() == 2) {
			cc->GetScheme()->ModReduceInternalInPlace(ctxtEnc, lbcrypto::BASE_NUM_LEVELS_TO_DROP);
		}
	}

	auto ctxtEnc_ = ctxtEnc->Clone();
	{
		lbcrypto::Plaintext result;
		cc->Decrypt(keys.secretKey, ctxtEnc, &result);

		std::cout << "Starting point after coeffs to slots:\n";
		std::cout << "Result " << result;
	}

	{
		FIDESlib::CKKS::Context& cc_       = GPUcc;
		cc_                                = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
		FIDESlib::CKKS::ContextData& GPUcc = *cc_;

		FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, 8, cc_);
		FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
		FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
		kskEval.Initialize(rawKskEval);
		GPUcc.AddEvalKey(std::move(kskEval));

		FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, ctxtEnc_);
		FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
		std::cout << "coefficients ";
		for (auto& i : GPUcc.GetCoeffsChebyshev())
			std::cout << i << " ";

		// coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475
		// 0.00428562 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264
		// 0.00211163 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128
		// -0.00695249 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494
		// -0.000989932 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05
		// -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10
		// -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
		std::cout << std::endl;

		std::cout << "coefficients ";
		for (auto& i : GPUcc.GetCoeffsChebyshev())
			std::cout << i << " ";

		ctxtEnc = ctxtEnc_->Clone();
		// coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475
		// 0.00428562 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264
		// 0.00211163 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128
		// -0.00695249 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494
		// -0.000989932 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05
		// -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10
		// -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
		std::cout << std::endl;
		// Evaluate Chebyshev series for the sine wave
		ctxtEnc = cc->EvalChebyshevSeries(ctxtEnc, GPUcc.GetCoeffsChebyshev(), -1.0, 1.0);

		/*{
				lbcrypto::Plaintext result;
				cc->Decrypt(keys.secretKey, ctxtEnc, &result);

				std::cout << "After Chebyshev:\n";
				std::cout << "Result " << result;
			}*/

		// Double-angle iterations
		if (true //(cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY) ||
			//(cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY)
		) {
			if (false // cryptoParams->GetScalingTechnique() != FIXEDMANUAL
			) {
				// algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
			}
			uint32_t numIter;
			// if (cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY)
			//     numIter = R_UNIFORM;
			// else
			//     numIter = R_SPARSE;

			numIter = GPUcc.GetDoubleAngleIts();
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE)->ApplyDoubleAngleIterations(ctxtEnc, numIter);
		}

		// scale the message back up after Chebyshev interpolation
		cc->GetScheme()->MultByIntegerInPlace(ctxtEnc, 1.0);

		cc->Decrypt(keys.secretKey, ctxtEnc, &result);

		std::cout << "After ApproxModEval CPU:\n";
		std::cout << "Result " << result;

		for (int batch : { 2 }) {
			fideslibParams.batch = batch;
			std::cout << "Batch " << batch << std::endl;
			GPUcc.batch = batch;
			cudaDeviceSynchronize();
			FIDESlib::CKKS::Ciphertext GPUct1(cc_);
			GPUct1.copy(GPUct1_);

			FIDESlib::CKKS::approxModReductionSparse(GPUct1, 1.0);

			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);

			{

				auto cResGPU = c2->Clone();
				GetOpenFHECipherText(cResGPU, raw_res1);
				lbcrypto::Plaintext resultGPU;
				cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
				std::cout << "Result GPU after ApproxModEval" << resultGPU;
				CudaCheckErrorMod;
				ASSERT_ERROR_OK(result, resultGPU);
				// ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
			}

			CudaCheckErrorMod;
		}
	}
}

TEST_P(OpenFHEBootstrapTest, LinearTransform) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 0.1, -0.1, -0.75, -0.5, -0.25 };;

	const int slots = 32;
	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT ? 2 : 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	lbcrypto::Plaintext result;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({ 1, 1 }, { 4, 4 }, slots);

	std::cout << "Generate keys" << std::endl;

	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	// coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562
	// 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163
	// 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249
	// 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932
	// 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05
	// -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10
	// -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
	std::cout << "Run bootstrap start" << std::endl;
	auto FHE    = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
	auto raised = c1->Clone(); // FHE->EvalBootstrapSetupOnly(c1, 1, 0);
	// cc->GetScheme()->ModReduceInternalInPlace(raised, 1);
	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, raised);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
	{
		lbcrypto::Plaintext result;
		cc->Decrypt(keys.secretKey, raised, &result);

		std::cout << "Before linear transform:\n";
		std::cout << "Result " << result;

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1_.store(raw_res1);
		auto cResGPU = c2->Clone();
		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;
		CudaCheckErrorMod;
		ASSERT_ERROR_OK(result, resultGPU);
		ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
	}
	/*
		{
			{
				auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(raised->GetKeyTag());
				auto conj = FHE->Conjugate(raised, evalKeyMap);
				cc->EvalAddInPlace(raised, conj);
			}
			lbcrypto::Plaintext result;
			cc->Decrypt(keys.secretKey, raised, &result);

			std::cout << "Before linear transform:\n";
			std::cout << "Result " << result;

			//GPUct1.addPt(GPUcc.GetBootPrecomputation(8).A[0]);
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);

			auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
			auto conj = FHE->Conjugate(cResGPU, evalKeyMap);
			cc->EvalAddInPlace(cResGPU, conj);
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU " << resultGPU;
			CudaCheckErrorMod;
			ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
		}
		*/
	std::cout << "Run linear transform" << std::endl;

	auto& plains = FHE->m_bootPrecomMap.at(slots)->m_U0hatTPre;
	auto ctxtEnc = FHE->EvalLinearTransform(plains, raised);

	auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
	auto conj       = FHE->Conjugate(ctxtEnc, evalKeyMap);
	cc->EvalAddInPlace(ctxtEnc, conj);

	//    lbcrypto::Plaintext result;
	cc->Decrypt(keys.secretKey, ctxtEnc, &result);

	std::cout << "After linear transform:\n";
	std::cout << "Result " << result;

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "GPUs " << GPUcc.GPUid.size() << std::endl;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		std::cout << "Run linear transform GPU" << std::endl;
		/*
		FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
		FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
		kskEval.Initialize(rawKskEval);
		*/

		CudaCheckErrorMod;
		FIDESlib::CKKS::EvalLinearTransform(GPUct1, slots, false);
		CudaCheckErrorMod;
		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);

			{
				auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
				auto conj       = FHE->Conjugate(cResGPU, evalKeyMap);
				cc->EvalAddInPlace(cResGPU, conj);
			}

			/*
			while (ctxtEnc->GetNoiseScaleDeg() > 1) {
				cc->ModReduceInPlace(ctxtEnc);
			}
*/
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU after LT" << resultGPU;
			CudaCheckErrorMod;
			ASSERT_ERROR_OK(result, resultGPU);
			ASSERT_EQ_CIPHERTEXT(ctxtEnc, cResGPU);
		}

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEBootstrapTest, CoeffsToSlots) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 0.1, -0.1, -0.75, -0.5, -0.25 };

	// Encoding as plaintexts
	int slots                 = GPUcc.N / 2; // cc->GetRingDimension() / 2;
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT ? 2 : 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	lbcrypto::Plaintext result;
	std::cout << "Setup Bootstrap" << std::endl;
	// {0, 0} lets OpenFHE choose the baby-step split, which is what `thorfhe.bench` passes and what
	// the SlotsToCoeffs test below already uses. Pinned to {16, 16} this exercised a decomposition
	// with a different diagonal count and rotation set than the one that runs, so a green result
	// here said nothing about the benchmark.
	cc->EvalBootstrapSetup({ 3, 3 }, { 0, 0 }, slots);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	// coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562
	// 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163
	// 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249
	// 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932
	// 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05
	// -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10
	// -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
	std::cout << "Run bootstrap start" << std::endl;
	auto FHE    = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
	auto raised = c1->Clone(); // FHE->EvalBootstrapSetupOnly(c1, 1, 0);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, raised);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
	{
		lbcrypto::Plaintext result;
		cc->Decrypt(keys.secretKey, raised, &result);

		std::cout << "Before linear transform:\n";
		std::cout << "Result " << result;

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1_.store(raw_res1);
		auto cResGPU = c2->Clone();
		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;
		CudaCheckErrorMod;
		ASSERT_ERROR_OK(result, resultGPU);
		ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
	}
	/*
		{
			{
				auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(raised->GetKeyTag());
				auto conj = FHE->Conjugate(raised, evalKeyMap);
				cc->EvalAddInPlace(raised, conj);
			}
			lbcrypto::Plaintext result;
			cc->Decrypt(keys.secretKey, raised, &result);

			std::cout << "Before linear transform:\n";
			std::cout << "Result " << result;

			//GPUct1.addPt(GPUcc.GetBootPrecomputation(8).A[0]);
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);

			auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
			auto conj = FHE->Conjugate(cResGPU, evalKeyMap);
			cc->EvalAddInPlace(cResGPU, conj);
			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU " << resultGPU;
			CudaCheckErrorMod;
			ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
		}
		*/
	std::cout << "Run CoeffToSlot" << std::endl;

	auto ctxtEnc = FHE->EvalCoeffsToSlots(FHE->m_bootPrecomMap.at(slots)->m_U0hatTPreFFT, raised);

	cc->RescaleInPlace(ctxtEnc);

	auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
	auto conj       = FHE->Conjugate(ctxtEnc, evalKeyMap);
	cc->EvalAddInPlace(ctxtEnc, conj);

	//    lbcrypto::Plaintext result;
	cc->Decrypt(keys.secretKey, ctxtEnc, &result);

	std::cout << "After CoeffToSlot:\n";
	std::cout << "Result " << result;

	std::cout << "Run CoeffToSlot GPU" << std::endl;
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		CudaCheckErrorMod;
		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, false);

		CudaCheckErrorMod;
		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);

			{
				auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
				auto conj       = FHE->Conjugate(cResGPU, evalKeyMap);
				cc->EvalAddInPlace(cResGPU, conj);
			}

			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU after LT" << resultGPU;
			CudaCheckErrorMod;
			ASSERT_ERROR_OK(result, resultGPU);
		}

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEBootstrapTest, SlotsToCoeffs) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	// Encoding as plaintexts
	int slots                 = GPUcc.N / 2;
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, (GPUcc.rescaleTechnique == CKKS::FLEXIBLEAUTOEXT ? 2 : 1) + 3 + (7 + 6), nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	lbcrypto::Plaintext result;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({ 3, 3 }, { 0, 0 }, slots);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	// coefficients 0.154214 -0.00376715 0.16032 -0.00345397 0.177115 -0.00276197 0.199498 -0.0015928 0.217569 0.0001073 0.216004 0.00221714 0.176475 0.00428562
	// 0.0861745 0.00546403 -0.046668 0.00473469 -0.177127 0.00162051 -0.227031 -0.00281458 -0.131231 -0.00563456 0.0788184 -0.00378689 0.232264 0.00211163
	// 0.139855 0.00593656 -0.139185 0.00185807 -0.232544 -0.00541038 0.0568406 -0.00352272 0.256679 0.00550297 -0.0733344 0.00278103 -0.249128 -0.00695249
	// 0.212888 0.00178101 0.088761 0.00559572 -0.319372 -0.00875394 0.347488 0.00753783 -0.251165 -0.00472857 0.139705 0.00236725 -0.0636494 -0.000989932
	// 0.0245978 0.000355532 -0.0082485 -0.000111762 0.00243906 3.11804e-05 -0.000643735 -7.8036e-06 0.0001531 1.76708e-06 -3.30668e-05
	// -3.64609e-07 6.5277e-06 6.89578e-08 -1.18428e-06 -1.20151e-08 1.98393e-07 1.9372e-09 -3.08154e-08 -2.90138e-10 4.45409e-09 4.05051e-11 -6.01049e-10
	// -5.28733e-12 7.59432e-11 6.46796e-13 -9.00812e-12 -7.43969e-14 1.00574e-12 8.17012e-15 -1.06117e-13 -8.95975e-16 1.14216e-14
	std::cout << "Run bootstrap start" << std::endl;
	auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
	// auto raised = FHE->EvalBootstrapNoStC(c1, 1, 0);
	// cc->GetScheme()->ModReduceInternalInPlace(c1, 3 + (7 + 6));
	auto raised                        = c1->Clone();
	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, raised);
	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);
	{
		lbcrypto::Plaintext result;
		cc->Decrypt(keys.secretKey, raised, &result);

		std::cout << "Before SlotsToCoeffs:\n";
		std::cout << "Result " << result;

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1_.store(raw_res1);
		auto cResGPU = c2->Clone();
		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
		std::cout << "Result GPU " << resultGPU;
		CudaCheckErrorMod;
		ASSERT_ERROR_OK(result, resultGPU);
		ASSERT_EQ_CIPHERTEXT(raised, cResGPU);
	}

	std::cout << "Run SlotsToCoeffs" << std::endl;

	auto ctxtEnc = FHE->EvalSlotsToCoeffs(FHE->m_bootPrecomMap.at(slots)->m_U0PreFFT, raised);

	cc->RescaleInPlace(ctxtEnc);
	if (slots < GPUcc.N / 2) {
		auto conj = cc->EvalRotate(ctxtEnc, slots);
		cc->EvalAddInPlace(ctxtEnc, conj);
	}
	//    lbcrypto::Plaintext result;
	cc->Decrypt(keys.secretKey, ctxtEnc, &result);

	std::cout << "After SlotsToCoeffs:\n";
	std::cout << "Result " << result;

	std::cout << "Run SlotsToCoeffs GPU" << std::endl;
	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();
		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		FIDESlib::CKKS::EvalCoeffsToSlots(GPUct1, slots, true);

		{
			FIDESlib::CKKS::RawCipherText raw_res1;
			GPUct1.store(raw_res1);
			auto cResGPU = c2->Clone();
			GetOpenFHECipherText(cResGPU, raw_res1);

			if (slots < GPUcc.N / 2) {
				auto conj = cc->EvalRotate(cResGPU, slots);
				cc->EvalAddInPlace(cResGPU, conj);
			}

			lbcrypto::Plaintext resultGPU;
			cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);
			std::cout << "Result GPU after SlotsToCoeffs" << resultGPU;
			CudaCheckErrorMod;
			ASSERT_ERROR_OK(result, resultGPU);
		}

		CudaCheckErrorMod;
	}
}

/*
TEST_P(OpenFHEBootstrapTest, OpenFHEBootstrapCPUsetup) {
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)
				->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	cc->EvalMultKeyGen(keys.secretKey);

	int slots = 1 << 4;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({2, 2}, {2, 2}, slots);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	///// PROBAR /////
	std::vector<double> x1 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};
	std::vector<double> x2 = {0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0};

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, raw_param.L - 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	auto& key = keys.publicKey;
	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	auto cAdd = cc->EvalBootstrap(c1);

	lbcrypto::Plaintext result;
	std::cout << cAdd->GetLevel() << "\n";
	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Result " << result;

	FIDESlib::CKKS::Context& cc_ = GPUcc; cc_ = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, GPUcc);

	///////////////////////////////////////////////////////////7777

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc.AddEvalKey(std::move(kskEval));

	//auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
	//c1 = FHE->EvalBootstrapSetupOnly(c1, 1, 0);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct_o(cc_, raw1);

	if constexpr (true) {
		cudaDeviceSynchronize();
		std::cout << "Initial ";
		for (auto& j : GPUct_o.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb)
				SWITCH(i, printThisLimb(1));
		}
		cudaDeviceSynchronize();
	}
	CudaCheckErrorMod;

	for (int batch : FIDESlib::Testing::batch_configs) {
	fideslibParams.batch = batch;
	std::cout << "Batch " << batch << std::endl;

	GPUcc.batch = batch;
	cudaDeviceSynchronize();

	FIDESlib::CKKS::Ciphertext GPUct1(cc_);
	for (int i = 0; i < 1; ++i) {
		GPUct1.copy(GPUct_o);

		cudaDeviceSynchronize();

		FIDESlib::CKKS::BootstrapCPUraise(GPUct1, slots, cc, keys, false);
	}

	FIDESlib::CKKS::RawCipherText raw_res1;
	GPUct1.store(raw_res1);
	auto cResGPU(c2);

	GetOpenFHECipherText(cResGPU, raw_res1);
	lbcrypto::Plaintext resultGPU;
	cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

	std::cout << "Result GPU " << resultGPU;

	CudaCheckErrorMod;
	ASSERT_ERROR_OK(result, resultGPU);
	//ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

	CudaCheckErrorMod;
}
}
*/

TEST_P(OpenFHEBootstrapTest, OpenFHEBootstrap) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	cc->EvalMultKeyGen(keys.secretKey);

	int slots = 1 << 5;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({ 2, 2 }, { 0, 0 }, slots);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, raw_param.L - 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	// auto cAdd = cc->EvalBootstrap(c1);
	auto cAdd = c1->Clone();

	lbcrypto::Plaintext result;
	std::cout << cAdd->GetLevel() << "\n";
	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Result " << result;

	FIDESlib::CKKS::Context& cc_       = GPUcc;
	cc_                                = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	///////////////////////////////////////////////////////////

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct_o(cc_, raw1);

	if constexpr (true) {
		cudaDeviceSynchronize();
		std::cout << "Initial ";
		for (auto& j : GPUct_o.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb)
				SWITCH(i, printThisLimb(1));
		}
		cudaDeviceSynchronize();
	}
	CudaCheckErrorMod;

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;

		GPUcc.batch = batch;
		cudaDeviceSynchronize();

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		for (int i = 0; i < 1; ++i) {
			GPUct1.copy(GPUct_o);

			cudaDeviceSynchronize();

			FIDESlib::CKKS::Bootstrap(GPUct1, slots, false);
		}

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c2);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;

		CudaCheckErrorMod;
		ASSERT_ERROR_OK(result, resultGPU);
		// ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEBootstrapTest, OpenFHEBootstrapManualPrescale) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

	cc->EvalMultKeyGen(keys.secretKey);

	int slots = 1 << 4;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({ 2, 2 }, { 2, 2 }, slots);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, raw_param.L - 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	auto cAdd = cc->EvalBootstrap(c1);

	lbcrypto::Plaintext result;
	std::cout << cAdd->GetLevel() << "\n";
	cc->Decrypt(keys.secretKey, cAdd, &result);

	std::cout << "Result " << result;

	FIDESlib::CKKS::Context& cc_       = GPUcc;
	cc_                                = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc = *cc_;

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	///////////////////////////////////////////////////////////7777

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc.AddEvalKey(std::move(kskEval));

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);
	FIDESlib::CKKS::Ciphertext GPUct_o(cc_, raw1);

	if constexpr (true) {
		cudaDeviceSynchronize();
		std::cout << "Initial ";
		for (auto& j : GPUct_o.c0.GPU) {
			cudaSetDevice(j.device);
			for (auto& i : j.limb)
				SWITCH(i, printThisLimb(1));
		}
		cudaDeviceSynchronize();
	}
	CudaCheckErrorMod;

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;

		GPUcc.batch = batch;
		cudaDeviceSynchronize();

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		for (int i = 0; i < 4; ++i) {
			GPUct1.copy(GPUct_o);
			cudaDeviceSynchronize();
			GPUct1.dropToLevel(1);
			GPUct1.multScalar(CKKS::GetPreScaleFactor(cc_, slots), true);
			FIDESlib::CKKS::Bootstrap(GPUct1, slots, true);
		}

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c2);

		GetOpenFHECipherText(cResGPU, raw_res1);
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;

		CudaCheckErrorMod;
		ASSERT_ERROR_OK(result, resultGPU);
		// ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEBootstrapTest, OpenFHEBootstrapLT) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	int slots = 32;
	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	{
		lbcrypto::Plaintext result;
		cc->Decrypt(keys.secretKey, c1, &result);

		std::cout << "Result " << result;
	}

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);

	lbcrypto::Plaintext result;
	std::cout << "Setup Bootstrap" << std::endl;
	cc->EvalBootstrapSetup({ 1, 1 }, { 4, 4 }, slots, 0, true, false);

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	if (1) {
		// auto cAdd = cc->EvalBootstrap(c1);
		auto cAdd = c1->Clone();
		/*{
			cc->RescaleInPlace(cAdd);
			auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cAdd->GetKeyTag());
			auto conj = FHE->Conjugate(cAdd, evalKeyMap);
		}*/

		std::cout << cAdd->GetLevel() << "\n";
		cc->Decrypt(keys.secretKey, cAdd, &result);
		std::cout << "Result " << result;
	}
	///////////////////////////////////////////////////////////7777

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc.AddEvalKey(std::move(kskEval));

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;

		cudaDeviceSynchronize();

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);

		for (int i = 0; i < 4; ++i) {
			GPUct1.copy(GPUct1_);

			CKKS::Ciphertext::clearOpRecord();

			FIDESlib::CKKS::Bootstrap(GPUct1, slots, false);

			CKKS::Ciphertext::printOpRecord();
		}

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c2);

		GetOpenFHECipherText(cResGPU, raw_res1);

		auto FHE = std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(cc->GetScheme()->m_FHE);
		/*
		{
			auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(cResGPU->GetKeyTag());
			auto conj = FHE->Conjugate(cResGPU, evalKeyMap);
			cc->EvalAddInPlace(cResGPU, conj);
		}
		*/
		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Result GPU " << resultGPU;

		CudaCheckErrorMod;
		ASSERT_ERROR_OK(result, resultGPU);

		// ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

		CudaCheckErrorMod;
	}
}

TEST_P(OpenFHEBootstrapTest, OpenFHEBootstrapDense) {
	CKKS::DeregisterAllContexts();
	for (auto& i : cached_cc) {
		i.second.first->ClearEvalAutomorphismKeys();
		i.second.first->ClearEvalMultKeys();
		if (std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE))
			std::dynamic_pointer_cast<lbcrypto::FHECKKSRNS>(i.second.first->GetScheme()->m_FHE)->m_bootPrecomMap.clear();
	}
	// Enable the features that you wish to use
	cc->Enable(lbcrypto::PKE);
	cc->Enable(lbcrypto::KEYSWITCH);
	cc->Enable(lbcrypto::LEVELEDSHE);
	cc->Enable(lbcrypto::ADVANCEDSHE);
	cc->Enable(lbcrypto::FHE);
	std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;
	cc->EvalMultKeyGen(keys.secretKey);

	FIDESlib::CKKS::RawParams raw_param = FIDESlib::CKKS::GetRawParams(cc, bootConfig());
	FIDESlib::CKKS::Context& cc_        = GPUcc;
	cc_                                 = CKKS::GenCryptoContextGPU(fideslibParams.adaptTo(raw_param), devices);
	FIDESlib::CKKS::ContextData& GPUcc  = *cc_;
	GPUcc.batch                         = 100;
	///// PROBAR /////
	std::vector<double> x1 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };
	std::vector<double> x2 = { 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0 };

	int slots = GPUcc.N / 2;
	// Encoding as plaintexts
	lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, GPUcc.L - 1, nullptr, slots);
	lbcrypto::Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);

	std::cout << "Input x1: " << ptxt1 << std::endl;

	// Encrypt the encoded vectors
	auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
	auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

	FIDESlib::CKKS::RawCipherText raw1 = FIDESlib::CKKS::GetRawCipherText(cc, c1);

	FIDESlib::CKKS::Ciphertext GPUct1_(cc_, raw1);

	lbcrypto::Plaintext result;
	std::cout << "Setup Bootstrap" << std::endl;
	// cc->EvalBootstrapSetup({5, 5}, {0, 0}, slots);

	cc->EvalBootstrapSetup(
		{ 3, 3 },
		{ 16, 16 },
		slots,
		0,
		true,
		false,
		lbcrypto::GetMultiplicativeDepthByCoeffVector(GPUcc.GetCoeffsChebyshev(), false) + GPUcc.GetDoubleAngleIts());

	std::cout << "Generate keys" << std::endl;
	cc->EvalBootstrapKeyGen(keys.secretKey, slots);

	FIDESlib::CKKS::AddBootstrapPrecomputation(cc, keys, slots, cc_);

	if (1) {
		auto cAdd = cc->EvalBootstrap(c1);

		std::cout << cAdd->GetLevel() << "\n";
		cc->Decrypt(keys.secretKey, cAdd, &result);

		std::cout << "Result " << result;
	} else {
		std::cout << "Skipping CPU" << std::endl;
	}
	///////////////////////////////////////////////////////////7777

	FIDESlib::CKKS::KeySwitchingKey kskEval(cc_);
	FIDESlib::CKKS::RawKeySwitchKey rawKskEval = FIDESlib::CKKS::GetEvalKeySwitchKey(keys);
	kskEval.Initialize(rawKskEval);
	GPUcc.AddEvalKey(std::move(kskEval));

	for (int batch : FIDESlib::Testing::batch_configs) {
		fideslibParams.batch = batch;
		std::cout << "Batch " << batch << std::endl;
		GPUcc.batch = batch;
		cudaDeviceSynchronize();

		FIDESlib::CKKS::Ciphertext GPUct1(cc_);
		GPUct1.copy(GPUct1_);

		// FIDESlib::CKKS::Bootstrap(GPUct1, slots, false);
		// FIDESlib::CKKS::BootstrapCPUraise(GPUct1, slots, cc, keys, false);

		for (int i = 0; i < 3; ++i) {
			GPUct1.copy(GPUct1_);

			for (int j = 0; j < 10; ++j) {
				CKKS::Ciphertext GPUct_aux(cc_);
				GPUct_aux.copy(GPUct1);
				cudaDeviceSynchronize();
				FIDESlib::CKKS::Bootstrap(GPUct_aux, slots, false);
				cudaDeviceSynchronize();
				GPUct_aux.dropToLevel(2);
				GPUct1.copy(GPUct_aux);
			}
		}

		FIDESlib::CKKS::RawCipherText raw_res1;
		GPUct1.store(raw_res1);
		auto cResGPU(c2);

		GetOpenFHECipherText(cResGPU, raw_res1);

		lbcrypto::Plaintext resultGPU;
		cc->Decrypt(keys.secretKey, cResGPU, &resultGPU);

		std::cout << "Levels: " << cResGPU->GetLevel() << std::endl;
		std::cout << "Result GPU " << resultGPU;

		CudaCheckErrorMod;
		ASSERT_ERROR_OK(result, resultGPU);

		// ASSERT_EQ_CIPHERTEXT(cAdd, cResGPU);

		CudaCheckErrorMod;
	}
}

INSTANTIATE_TEST_SUITE_P(OpenFHEBootstrapTests, OpenFHEBootstrapTest, testing::Values(TTALL64BOOTTHOR));
} // namespace FIDESlib::Testing