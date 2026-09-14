// pybind11 bindings for the fideslib API.
//
// Design: the fideslib CryptoContext already dispatches every operation to OpenFHE (CPU) when the device
// list is empty and to CUDA otherwise, so one binding serves both backends. numpy arrays are the memory
// bridge for messages (encode/decrypt); ciphertexts stay opaque handles owned by the context.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fideslib.hpp>

#include <algorithm>
#include <complex>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace fideslib;

using CC = CryptoContext<DCRTPoly>;
using CT = Ciphertext<DCRTPoly>;
using PT = Plaintext;
using LPT = LightPlaintext;

namespace {

std::vector<std::complex<double>> ToComplexVector(const py::array& arr) {
	if (py::isinstance<py::array_t<std::complex<double>>>(arr)) {
		auto a = arr.cast<py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>>();
		return std::vector<std::complex<double>>(a.data(), a.data() + a.size());
	}
	auto a = arr.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
	std::vector<std::complex<double>> out(a.size());
	for (py::ssize_t i = 0; i < a.size(); ++i)
		out[i] = std::complex<double>(a.data()[i], 0.0);
	return out;
}

py::array_t<std::complex<double>> ToNumpy(const std::vector<std::complex<double>>& v) {
	py::array_t<std::complex<double>> out(v.size());
	std::copy(v.begin(), v.end(), out.mutable_data());
	return out;
}

} // namespace

PYBIND11_MODULE(_core, m) {
	m.doc() = "fideslib CKKS bindings (CPU/OpenFHE or CUDA backend, selected per context)";

	py::enum_<PKESchemeFeature>(m, "PKESchemeFeature")
	  .value("PKE", PKE)
	  .value("KEYSWITCH", KEYSWITCH)
	  .value("LEVELEDSHE", LEVELEDSHE)
	  .value("ADVANCEDSHE", ADVANCEDSHE)
	  .value("FHE", FHE)
	  .export_values();
	py::enum_<ScalingTechnique>(m, "ScalingTechnique")
	  .value("FIXEDMANUAL", FIXEDMANUAL)
	  .value("FIXEDAUTO", FIXEDAUTO)
	  .value("FLEXIBLEAUTO", FLEXIBLEAUTO)
	  .value("FLEXIBLEAUTOEXT", FLEXIBLEAUTOEXT)
	  .export_values();
	py::enum_<KeySwitchTechnique>(m, "KeySwitchTechnique").value("HYBRID", HYBRID).export_values();
	py::enum_<SecretKeyDist>(m, "SecretKeyDist")
	  .value("UNIFORM_TERNARY", UNIFORM_TERNARY)
	  .value("SPARSE_TERNARY", SPARSE_TERNARY)
	  .value("SPARSE_ENCAPSULATED", SPARSE_ENCAPSULATED)
	  .export_values();
	py::enum_<SecurityLevel>(m, "SecurityLevel")
	  .value("HEStd_128_classic", HEStd_128_classic)
	  .value("HEStd_192_classic", HEStd_192_classic)
	  .value("HEStd_256_classic", HEStd_256_classic)
	  .value("HEStd_NotSet", HEStd_NotSet)
	  .export_values();
	py::enum_<CKKSDataType>(m, "CKKSDataType")
	  .value("REAL", REAL)
	  .value("COMPLEX", COMPLEX)
	  .export_values();

	py::class_<CCParams<CryptoContextCKKSRNS>>(m, "CCParams")
	  .def(py::init<>())
	  .def("SetMultiplicativeDepth", &CCParams<CryptoContextCKKSRNS>::SetMultiplicativeDepth)
	  .def("SetScalingModSize", &CCParams<CryptoContextCKKSRNS>::SetScalingModSize)
	  .def("SetBatchSize", &CCParams<CryptoContextCKKSRNS>::SetBatchSize)
	  .def("SetRingDim", &CCParams<CryptoContextCKKSRNS>::SetRingDim)
	  .def("SetScalingTechnique", &CCParams<CryptoContextCKKSRNS>::SetScalingTechnique)
	  .def("SetNumLargeDigits", &CCParams<CryptoContextCKKSRNS>::SetNumLargeDigits)
	  .def("SetFirstModSize", &CCParams<CryptoContextCKKSRNS>::SetFirstModSize)
	  .def("SetDigitSize", &CCParams<CryptoContextCKKSRNS>::SetDigitSize)
	  .def("SetKeySwitchTechnique", &CCParams<CryptoContextCKKSRNS>::SetKeySwitchTechnique)
	  .def("SetSecretKeyDist", &CCParams<CryptoContextCKKSRNS>::SetSecretKeyDist)
	  .def("SetSecurityLevel", &CCParams<CryptoContextCKKSRNS>::SetSecurityLevel)
	  .def("SetCKKSDataType", &CCParams<CryptoContextCKKSRNS>::SetCKKSDataType)
	  .def("SetDevices", [](CCParams<CryptoContextCKKSRNS>& p, std::vector<int> devs) { p.SetDevices(std::move(devs)); })
	  .def("SetPlaintextAutoload", &CCParams<CryptoContextCKKSRNS>::SetPlaintextAutoload)
	  .def("SetCiphertextAutoload", &CCParams<CryptoContextCKKSRNS>::SetCiphertextAutoload)
	  .def("GetMultiplicativeDepth", &CCParams<CryptoContextCKKSRNS>::GetMultiplicativeDepth)
	  .def("GetBatchSize", &CCParams<CryptoContextCKKSRNS>::GetBatchSize);

	py::class_<PublicKeyImpl<DCRTPoly>, PublicKey<DCRTPoly>>(m, "PublicKey");
	py::class_<PrivateKeyImpl<DCRTPoly>, PrivateKey<DCRTPoly>>(m, "PrivateKey");
	py::class_<KeyPair<DCRTPoly>>(m, "KeyPair")
	  .def_readonly("publicKey", &KeyPair<DCRTPoly>::publicKey)
	  .def_readonly("secretKey", &KeyPair<DCRTPoly>::secretKey);

	py::class_<PlaintextImpl, PT>(m, "Plaintext")
	  .def_readonly("loaded", &PlaintextImpl::loaded)
	  .def("SetLength", &PlaintextImpl::SetLength)
	  .def("GetLevel", &PlaintextImpl::GetLevel)
	  .def("GetLogPrecision", &PlaintextImpl::GetLogPrecision)
	  .def("GetCKKSPackedValue", [](const PlaintextImpl& p) { return ToNumpy(p.GetCKKSPackedValue()); })
	  .def("GetRealPackedValue", [](const PlaintextImpl& p) {
		  auto v = p.GetRealPackedValue();
		  py::array_t<double> out(v.size());
		  std::copy(v.begin(), v.end(), out.mutable_data());
		  return out;
	  });

	py::class_<LightPlaintextImpl, LPT>(m, "LightPlaintext")
	  .def_readonly("scale", &LightPlaintextImpl::scale)
	  .def_readonly("slots", &LightPlaintextImpl::slots)
	  .def_readonly("level_hint", &LightPlaintextImpl::level_hint)
	  .def("nbytes", &LightPlaintextImpl::Bytes, "Bytes the compact form occupies")
	  .def("save", &LightPlaintextImpl::Save, py::arg("path"))
	  .def_static("load", &LightPlaintextImpl::Load, py::arg("path"))
	  .def("coefficients", [](const LightPlaintextImpl& lp) {
		  py::array_t<int64_t> out(lp.coeffs.size());
		  std::copy(lp.coeffs.begin(), lp.coeffs.end(), out.mutable_data());
		  return out;
	  });

	py::class_<CiphertextImpl<DCRTPoly>, CT>(m, "Ciphertext")
	  .def("Clone", &CiphertextImpl<DCRTPoly>::Clone)
	  .def("GetLevel", &CiphertextImpl<DCRTPoly>::GetLevel)
	  .def_readonly("loaded", &CiphertextImpl<DCRTPoly>::loaded);

	py::class_<CryptoContextImpl<DCRTPoly>, CC>(m, "CryptoContext")
	  // ---- setup ----
	  .def("Enable", py::overload_cast<PKESchemeFeature>(&CryptoContextImpl<DCRTPoly>::Enable))
	  .def("GetRingDimension", &CryptoContextImpl<DCRTPoly>::GetRingDimension)
	  .def("SetDevices", &CryptoContextImpl<DCRTPoly>::SetDevices)
	  .def("KeyGen", &CryptoContextImpl<DCRTPoly>::KeyGen)
	  .def("EvalMultKeyGen", &CryptoContextImpl<DCRTPoly>::EvalMultKeyGen)
	  .def("EvalRotateKeyGen", &CryptoContextImpl<DCRTPoly>::EvalRotateKeyGen)
	  .def("EvalConjugateKeyGen", &CryptoContextImpl<DCRTPoly>::EvalConjugateKeyGen)
	  .def("SetRotationKeyLevels", &CryptoContextImpl<DCRTPoly>::SetRotationKeyLevels, py::arg("max_remaining_levels"),
		"index -> maximum remaining levels the rotation key is used at (THOR create_fixed_rotation_key level)")
	  .def("EvalBootstrapSetup", &CryptoContextImpl<DCRTPoly>::EvalBootstrapSetup, py::arg("levelBudget") = std::vector<uint32_t>{ 5, 4 },
		py::arg("dim1") = std::vector<uint32_t>{ 0, 0 }, py::arg("slots") = 0, py::arg("correctionFactor") = 0, py::arg("precompute") = true,
		py::arg("btsfirstboot") = false)
	  .def("EvalBootstrapKeyGen", &CryptoContextImpl<DCRTPoly>::EvalBootstrapKeyGen)
	  .def("LoadContext", &CryptoContextImpl<DCRTPoly>::LoadContext)
	  .def_readwrite("truncate_keys", &CryptoContextImpl<DCRTPoly>::truncate_keys)
	  .def_readwrite("key_level_margin", &CryptoContextImpl<DCRTPoly>::key_level_margin)
	  .def_readwrite("allow_key_grow", &CryptoContextImpl<DCRTPoly>::allow_key_grow)
	  .def("GetKeyDeviceBytes", &CryptoContextImpl<DCRTPoly>::GetKeyDeviceBytes)
	  .def("GetGrownKeyCount", &CryptoContextImpl<DCRTPoly>::GetGrownKeyCount)
	  // ---- encoding / encryption (numpy bridge) ----
	  .def(
		"MakeLightPlaintext",
		[](CryptoContextImpl<DCRTPoly>& cc, const py::array& value, uint32_t slots, int32_t level_hint) {
			return cc.MakeLightPlaintext(ToComplexVector(value), slots, level_hint);
		},
		py::arg("value"), py::arg("slots") = 0, py::arg("level_hint") = -1,
		"Encode into the compact coefficient form (THOR encode_to_light_plaintext)")
	  .def("ExpandLightPlaintext", &CryptoContextImpl<DCRTPoly>::ExpandLightPlaintext, py::arg("light"), py::arg("level"))
	  .def("ClearLightPlaintextCache", &CryptoContextImpl<DCRTPoly>::ClearLightPlaintextCache)
	  .def("TrimAuxiliaryPolys", &CryptoContextImpl<DCRTPoly>::TrimAuxiliaryPolys, py::arg("keep") = 0)
	  .def("GetAuxiliaryPolyCount", &CryptoContextImpl<DCRTPoly>::GetAuxiliaryPolyCount)
	  .def("GetDeviceMemory", &CryptoContextImpl<DCRTPoly>::GetDeviceMemory)
	  .def("GetLightPlaintextCacheSize", &CryptoContextImpl<DCRTPoly>::GetLightPlaintextCacheSize)
	  .def_readwrite("light_plaintext_cache_capacity", &CryptoContextImpl<DCRTPoly>::light_plaintext_cache_capacity)
	  .def("GetConsumedLevels", &CryptoContextImpl<DCRTPoly>::GetConsumedLevels)
	  .def("GetNoiseLevel", &CryptoContextImpl<DCRTPoly>::GetNoiseLevel)
	  .def(
		"MakeCKKSPackedPlaintext",
		[](CryptoContextImpl<DCRTPoly>& cc, const py::array& value, size_t noiseScaleDeg, uint32_t level, uint32_t slots) {
			return cc.MakeCKKSPackedPlaintext(ToComplexVector(value), noiseScaleDeg, level, nullptr, slots);
		},
		py::arg("value"), py::arg("noiseScaleDeg") = 1, py::arg("level") = 0, py::arg("slots") = 0)
	  .def("Encrypt", [](CryptoContextImpl<DCRTPoly>& cc, const PublicKey<DCRTPoly>& pk, PT& pt) { return cc.Encrypt(pk, pt); })
	  .def("EncryptSecret", [](CryptoContextImpl<DCRTPoly>& cc, const PrivateKey<DCRTPoly>& sk, PT& pt) { return cc.Encrypt(sk, pt); })
	  .def(
		"Decrypt",
		[](CryptoContextImpl<DCRTPoly>& cc, const PrivateKey<DCRTPoly>& sk, CT& ct, size_t length) {
			PT pt;
			cc.Decrypt(sk, ct, &pt);
			if (length)
				pt->SetLength(length);
			return ToNumpy(pt->GetCKKSPackedValue());
		},
		py::arg("sk"), py::arg("ct"), py::arg("length") = 0, "Decrypt to a complex128 numpy array")
	  // ---- arithmetic ----
	  .def("EvalAdd", py::overload_cast<const CT&, const CT&>(&CryptoContextImpl<DCRTPoly>::EvalAdd))
	  .def("EvalAddPt", py::overload_cast<const CT&, PT&>(&CryptoContextImpl<DCRTPoly>::EvalAdd))
	  .def("EvalAddLightPt", py::overload_cast<const CT&, const LPT&>(&CryptoContextImpl<DCRTPoly>::EvalAdd))
	  .def("EvalAddScalar", py::overload_cast<const CT&, double>(&CryptoContextImpl<DCRTPoly>::EvalAdd))
	  .def("EvalAddInPlace", py::overload_cast<CT&, const CT&>(&CryptoContextImpl<DCRTPoly>::EvalAddInPlace))
	  .def("EvalSub", py::overload_cast<const CT&, const CT&>(&CryptoContextImpl<DCRTPoly>::EvalSub))
	  .def("EvalSubPt", py::overload_cast<const CT&, PT&>(&CryptoContextImpl<DCRTPoly>::EvalSub))
	  .def("EvalSubScalar", py::overload_cast<const CT&, double>(&CryptoContextImpl<DCRTPoly>::EvalSub))
	  .def("EvalScalarSub", py::overload_cast<double, const CT&>(&CryptoContextImpl<DCRTPoly>::EvalSub))
	  .def("EvalNegate", &CryptoContextImpl<DCRTPoly>::EvalNegate)
	  .def("EvalMult", py::overload_cast<const CT&, const CT&>(&CryptoContextImpl<DCRTPoly>::EvalMult))
	  .def("EvalMultPt", py::overload_cast<const CT&, PT&>(&CryptoContextImpl<DCRTPoly>::EvalMult))
	  .def("EvalMultLightPt", py::overload_cast<const CT&, const LPT&>(&CryptoContextImpl<DCRTPoly>::EvalMult))
	  .def("EvalMultScalar", py::overload_cast<const CT&, double>(&CryptoContextImpl<DCRTPoly>::EvalMult))
	  .def("EvalSquare", &CryptoContextImpl<DCRTPoly>::EvalSquare)
	  .def("EvalMultNoRelin", &CryptoContextImpl<DCRTPoly>::EvalMultNoRelin)
	  .def("EvalSquareNoRelin", &CryptoContextImpl<DCRTPoly>::EvalSquareNoRelin)
	  .def("EvalRelinearize", &CryptoContextImpl<DCRTPoly>::EvalRelinearize)
	  .def("EvalMultByI", &CryptoContextImpl<DCRTPoly>::EvalMultByI)
	  .def("EvalMultByInteger", &CryptoContextImpl<DCRTPoly>::EvalMultByInteger)
	  .def("EvalConjugate", &CryptoContextImpl<DCRTPoly>::EvalConjugate)
	  .def("EvalRotate", &CryptoContextImpl<DCRTPoly>::EvalRotate)
	  .def("Rescale", &CryptoContextImpl<DCRTPoly>::Rescale)
	  .def("EvalLevelReduce", &CryptoContextImpl<DCRTPoly>::EvalLevelReduce)
	  .def("GetRemainingLevels", &CryptoContextImpl<DCRTPoly>::GetRemainingLevels)
	  .def("EvalBootstrap", &CryptoContextImpl<DCRTPoly>::EvalBootstrap, py::arg("ct"), py::arg("numIterations") = 1, py::arg("precision") = 0,
		py::arg("prescaled") = false);

	m.def("GenCryptoContext", [](CCParams<CryptoContextCKKSRNS>& p) { return GenCryptoContext(p); });
}
