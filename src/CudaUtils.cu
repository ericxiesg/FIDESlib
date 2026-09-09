//
// Created by carlosad on 25/03/24.
//

#include <stdexcept>
#include <algorithm>
#include "CudaUtils.cuh"
#include <cassert>
#include <list>
#include <string>

#include "nvtx3/nvtx3.hpp"
#include <iostream>

// #include "driver_types.h"
#include "CKKS/Context.cuh"
#define DISABLE_STREAMS false

#include <cuda_runtime.h>

namespace FIDESlib {

extern std::vector<cudaDeviceProp> GPUprop;

struct my_domain {
	static constexpr char const* name{ "FIDESlib" };
};

nvtx3::domain const& D = nvtx3::domain::get<my_domain>();

std::map<std::string, std::pair<std::unique_ptr<nvtx3::unique_range_in<my_domain>>, int>> lifetimes_map;

void CudaNvtxStart(const std::string msg, NVTX_CATEGORIES cat, int val) {

	if (cat == FUNCTION) {
		using namespace nvtx3;
		int size = msg.size();
		const event_attributes attr{ msg,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ val },
		                             category{ static_cast<unsigned int>(cat) } };

		nvtxDomainRangePushEx_impl_init_v3(D, reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
		// nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
	} else if (cat == LIFETIME) {

		using namespace nvtx3;
		int size      = msg.size();
		auto& [r, i]  = lifetimes_map[msg];
		std::string m = std::to_string(i + 1) + std::string(" x ") + msg;
		const event_attributes attr{ m,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ i + 1 },
		                             category{ static_cast<unsigned int>(cat) } };
		i = i + 1;
		if (!r) {
			r = std::make_unique<unique_range_in<my_domain>>(attr);
		} else {
			*r = unique_range_in<my_domain>(attr);
		}
	}
	// nvtxRangePushA(msg.c_str());
}

void CudaNvtxStop(const std::string msg, NVTX_CATEGORIES cat) {
	if (cat == FUNCTION) {
		nvtxDomainRangePop(D);
	} else if (cat == LIFETIME) {
		using namespace nvtx3;
		int size = msg.size();

		auto& [r, i]  = lifetimes_map[msg];
		std::string m = std::to_string(i - 1) + std::string(" x ") + msg;
		const event_attributes attr{ m,
		                             rgb{ (uint8_t)(255 - 101 * msg[size / 6]), (uint8_t)(255 - 101 * msg[size * 3 / 6]),
		                                  (uint8_t)(255 - 101 * msg[size * 5 / 6]) },
		                             payload{ i - 1 },
		                             category{ static_cast<unsigned int>(cat) } };

		i = i - 1;
		if (i <= 0) {
			if (r) {
				r.reset();
			}
		} else {
			*r = unique_range_in<my_domain>(attr);
		}

		// nvtxRangePushEx(reinterpret_cast<const nvtxEventAttributes_t*>(&attr));
	}
}

int getNumDevices() {
	int d;
	cudaGetDeviceCount(&d);
	return d;
};

void CudaHostSync() {
	cudaDeviceSynchronize();
}

template <bool capture> void run_in_graph(cudaGraphExec_t& exec, Stream& s, std::function<void()> run) {
	cudaGraph_t graph;
	if constexpr (capture) {
		cudaStreamBeginCapture(s.ptr(), cudaStreamCaptureModeRelaxed);
		CudaCheckErrorModNoSync;
	}
	run();
	if constexpr (capture) {
		cudaStreamEndCapture(s.ptr(), &graph);
		if (!exec) {
			cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
			// cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
			CudaCheckErrorModNoSync;
		} else {
			if (cudaGraphExecUpdate(exec, graph, NULL) != cudaSuccess) {
				CudaCheckErrorModNoSync;
				// only instantiate a new graph if update fails
				cudaGraphExecDestroy(exec);
				cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagUseNodePriority);
				// cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
				CudaCheckErrorModNoSync;
			}
		}
		cudaGraphDestroy(graph);
		cudaGraphLaunch(exec, s.ptr());
	}
}

template void run_in_graph<false>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

template void run_in_graph<true>(cudaGraphExec_t& exec, Stream& s, std::function<void()> run);

/*
	void Stream::wait(const Event &ev) const {
		cudaStreamWaitEvent(ptr, ev.ptr());
	}
*/
void Stream::capture_begin() {
	CudaCheckErrorMod;
	std::cout << "Hello capture" << std::endl;
	cudaStreamCaptureStatus cap;
	cudaStreamIsCapturing(ptr(), &cap);

	CudaCheckErrorMod;
	if (cap == cudaStreamCaptureStatusNone) {
		std::cout << "None" << std::endl;
		cudaStreamBeginCapture(ptr(), cudaStreamCaptureModeGlobal);
	} else if (cap == cudaStreamCaptureStatusActive) {
		std::cout << "Fail: activo" << std::endl;
	} else if (cap == cudaStreamCaptureStatusInvalidated) {
		std::cout << "Fail: invalidado" << std::endl;
	} else {
		std::cout << "Fail" << std::endl;
	}
	CudaCheckErrorMod;
}

void Stream::capture_end() {
	cudaGraph_t graph;
	cudaStreamEndCapture(ptr(), &graph);
	CudaCheckErrorMod;
	cudaGraphExec_t graphExec;
	cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
	CudaCheckErrorMod;
	cudaGraphDestroy(graph);
	CudaCheckErrorMod;
	cudaGraphLaunch(graphExec, 0);
	cudaGraphExecDestroy(graphExec);

	cudaStreamSynchronize(0);
}

void Stream::record(bool external) {
	//if (ptr_ == 0)
	//	return;
	CudaCheckErrorModNoSync;
#if !DISABLE_STREAMS
	// cudaEventDestroy(ev);
	// cudaEventCreate(&ev, cudaEventDisableTiming);
	assert(ptr_ != nullptr);
	assert(ev != nullptr);
	cudaEventRecordWithFlags(ev, ptr_, external ? cudaEventRecordExternal : cudaEventRecordDefault);
	updated = true;
#endif
}

void Stream::wait_recorded(const Stream& s) {
	if (ptr_ == 0 || s.ptr_ == 0 || ptr_ == s.ptr_)
		return;
#if !DISABLE_STREAMS
	assert(s.updated);
	cudaStreamWaitEvent(ptr_, s.ev);
	this->updated = false;
#endif
}

void Stream::wait(Stream& s, bool external) {

#if !DISABLE_STREAMS
	// CudaCheckErrorModNoSync;
	if (/*ptr_ == 0 ||*/ s.ptr_ == 0 || ptr_ == s.ptr_) {
		updated = false;
		return;
	}
	//assert(ptr_ != nullptr);
	//assert(s.ptr_ != nullptr);
	assert(s.ev != nullptr);
	if (!s.updated) {
		assert(!external); // Has to be recorded in the origin graph
		CudaCheckErrorModNoSync;
		cudaEventRecordWithFlags(s.ev, s.ptr_, cudaEventRecordDefault);
		s.updated = true;
		CudaCheckErrorModNoSync;
	}
	CudaCheckErrorModNoSync;
	cudaStreamWaitEvent(ptr_, s.ev, external ? cudaEventWaitExternal : cudaEventWaitDefault);
	this->updated = false;
#endif
	CudaCheckErrorModNoSync;
}

void Stream::wait(cudaStream_t s) {

#if !DISABLE_STREAMS
	// CudaCheckErrorModNoSync;
	if (s == 0 || s == ptr_) {
		updated = false;
		return;
	}
	assert(ptr_ != nullptr);
	assert(ev != nullptr);
	CudaCheckErrorModNoSync;
	cudaEventRecordWithFlags(ev, s, cudaEventRecordDefault);
	updated = false;
	CudaCheckErrorModNoSync;

	CudaCheckErrorModNoSync;
	cudaStreamWaitEvent(ptr_, ev, cudaEventWaitDefault);
#endif
	CudaCheckErrorModNoSync;
}

int low  = -1;
int high = -1;

constexpr int POOL_SIZE = 37;
cudaStream_t stream_pool[MAXG][POOL_SIZE];

bool initPool() {
	int devs;
	cudaGetDeviceCount(&devs);
	for (int j = 0; j < devs; j++) {
		cudaSetDevice(j);
		for (int i = 0; i < POOL_SIZE; ++i) {
			cudaStreamCreateWithFlags(&stream_pool[j][i], 0);
		}
	}
	return true;
}

#define USEPOOL true
#if USEPOOL
bool initialized = initPool();
#else
bool initialized = false;
#endif

void Stream::init(int priority) {
	static int assigner_idx[MAXD] = { 0 };
	if (ptr_) {
		// free[ptr]++;
		cudaEventDestroy(ev);
		// cudaStreamDestroy(ptr_);
		ptr_ = nullptr;
		ev   = nullptr;
	}

#if !DISABLE_STREAMS
	if (high == -1) {
		cudaDeviceGetStreamPriorityRange(&low, &high);
	}

	// int prio = low + priority * ((high - low - 1)) / 100;

#if USEPOOL
	int dev;
	cudaGetDevice(&dev);
	ptr_ = stream_pool[dev][assigner_idx[dev]];
	assigner_idx[dev] += 1;
	if (assigner_idx[dev] >= POOL_SIZE)
		assigner_idx[dev] -= POOL_SIZE;
#else
	cudaStreamCreateWithPriority(&ptr_, 0 /*cudaStreamNonBlocking*/, priority);
	// cudaStreamCreateWithFlags(&ptr, cudaStreamNonBlocking);
#endif

	cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
	cudaEventCreate(&ev, cudaEventDisableTiming);
#else
	ptr_ = nullptr;
	ev   = nullptr;
#endif
	// free[ptr] = 0;
}

void Stream::initDefault() {
	ptr_    = 0;
	ev      = nullptr;
	updated = true;
}

// std::map<void *, int> free;

Stream::~Stream() {
	if (ptr_) {
		// free[ptr]++;
		// cudaStreamDestroy(ptr_);
		ptr_ = nullptr;
	}
	if (ev) {
		cudaEventDestroy(ev);
		ev = nullptr;
	}
}

Stream::Stream() = default;

Stream::Stream(Stream&& s) noexcept
	: ptr_(s.ptr_), ev(s.ev) {
	s.ptr_ = nullptr;
	s.ev   = nullptr;
}

std::vector<cudaDeviceProp> GPUprop;

void initGPUprop() {
	if (GPUprop.empty()) {
		int count = 0;
		cudaGetDeviceCount(&count);
		for (int i = 0; i < count; ++i) {
			GPUprop.emplace_back();
			cudaGetDeviceProperties(&GPUprop.back(), i);

			std::cout << "GPU " << i << ": " << GPUprop[i].name << "\n SMs: " << GPUprop[i].multiProcessorCount
				<< ", SharedMem: " << GPUprop[i].sharedMemPerMultiprocessor / 1024l << " KB, Blocks/SM: " << GPUprop[i].maxBlocksPerMultiProcessor
				<< ", Threads/SM: " << GPUprop[i].maxThreadsPerMultiProcessor << ", L2 size: " << GPUprop[i].l2CacheSize / (1024l * 1024l)
				<< " MB, Bus: " << (long long)GPUprop[i].memoryBusWidth

				<< "-bit" << std::endl;
		}
	}
}

std::mutex mempool_lock[MAXG];

std::map<int, std::vector<void*>> size_to_memory[MAXG];

/**
 * Every slab the pool has taken from the driver, so one can be given back.
 *
 * The pool is keyed by exact allocation size and, until now, never returned anything: a size class
 * that had grown could not lend its memory to one that had not. On a GV100 running a full THOR layer
 * that is what a run dies of - a 512 KiB slab request failing with 16 MiB free, all of it sitting in
 * other size classes' free lists.
 *
 * Reclaiming is only attempted when an allocation is about to fail, so the fast path is untouched.
 */
struct PoolSlab {
	char* base;
	size_t bytes;	 // the whole slab
	size_t block;	 // size class it was carved into
	size_t blocks;	 // how many blocks that is
};
std::vector<PoolSlab> pool_slabs[MAXG];

/**
 * Give back every slab all of whose blocks are currently free. Caller holds mempool_lock[id].
 * Returns how many bytes went back to the driver.
 */
static size_t ReclaimFreeSlabs(int id) {
	size_t reclaimed = 0;
	// Blocks handed out are still owned by kernels that may not have finished; only blocks sitting in
	// a free list are safe, and getting there already went through Stream::wait in GPUfree. Sync once
	// more before handing memory back, since this runs only on the failure path.
	cudaDeviceSynchronize();

	for (auto slab = pool_slabs[id].begin(); slab != pool_slabs[id].end();) {
		auto klass = size_to_memory[id].find((int)slab->block);
		if (klass == size_to_memory[id].end()) {
			++slab;
			continue;
		}
		std::vector<void*>& free_list = klass->second;
		const auto owns = [&slab](void* p) {
			char* c = (char*)p;
			return c >= slab->base && c < slab->base + slab->bytes;
		};
		size_t held = 0;
		for (void* p : free_list)
			held += owns(p) ? 1 : 0;

		if (held != slab->blocks) { // some blocks are still in use
			++slab;
			continue;
		}
		free_list.erase(std::remove_if(free_list.begin(), free_list.end(), owns), free_list.end());
		// Allocated with cudaMallocAsync, so it has to go back the same way.
		cudaFreeAsync(slab->base, s[id].ptr());
		reclaimed += slab->bytes;
		slab = pool_slabs[id].erase(slab);
	}
	if (reclaimed) {
		cudaStreamSynchronize(s[id].ptr()); // let the frees land before the caller retries
		std::cerr << "[FIDESlib] memory pool returned " << (reclaimed >> 20) << " MiB of wholly free slabs to the driver" << std::endl;
	}
	return reclaimed;
}

FIDESlib::Stream s[MAXG];

#define MEMPOOL true
// void* GPUmalloc(int id, int bytes, cudaStream_t stream, FIDESlib::CKKS::Context& cc) {
void* GPUmalloc(int id, int bytes, cudaStream_t stream, bool cache) {
	void* ptr = nullptr;

	uint64_t MBs = 1024;

	if (bytes < 64 * 1024) {
		int next_pow2 = 1024;
		while (next_pow2 < bytes) {
			next_pow2 *= 2;
		}
		bytes = next_pow2;
		cache = true;
		MBs   = bytes / 1024;
	}
#if MEMPOOL
	if (cache && (bytes & (bytes - 1)) == 0) {
		std::vector<void*>& free_limb = size_to_memory[id][bytes];

		if (s[id].ptr() == nullptr) {
			mempool_lock[id].lock();
			if (s[id].ptr() == nullptr) {
				s[id].init();
			}
			mempool_lock[id].unlock();
		}
		CudaCheckErrorModNoSync;
		if (free_limb.empty()) {
			// The pool is per size class and slabs are never returned to the driver, so late in a run
			// most of the "free" device memory is sitting in *other* size classes' free lists. Asking
			// for the full slab then fails even when nvidia-smi shows gigabytes free - which is
			// exactly the runtime OOM seen on a GV100 at depth 51: the first ciphertext copy needed a
			// limb size class keygen had never populated, and a 1 GiB slab request could not be met
			// with 8 GiB nominally free. Halve the request until it fits, down to a single block.
			uint64_t* base   = nullptr;
			uint64_t request = MBs * 1024 * 1024;
			const uint64_t floor_bytes = (uint64_t)bytes;
			cudaError_t err = cudaErrorMemoryAllocation;
			for (int attempt = 0; attempt < 2 && (err != cudaSuccess || base == nullptr); ++attempt) {
				request = MBs * 1024 * 1024;
				while (request >= floor_bytes) {
					err = cudaMallocAsync(&base, request, s[id].ptr());
					if (err == cudaSuccess)
						break;
					(void)cudaGetLastError(); // clear the sticky-free async error before retrying
					base = nullptr;
					if (request == floor_bytes)
						break;
					request = std::max(floor_bytes, request / 2);
				}
				// Nothing fits, not even one block: the memory is probably not gone, just parked in
				// other size classes. Hand back whatever is wholly free and try once more.
				if ((err != cudaSuccess || base == nullptr) && attempt == 0) {
					mempool_lock[id].lock();
					const size_t back = ReclaimFreeSlabs(id);
					mempool_lock[id].unlock();
					if (back == 0)
						break; // nothing to reclaim, a second pass would fail identically
					err = cudaErrorMemoryAllocation;
				}
			}
			if (err != cudaSuccess || base == nullptr) {
				size_t free_bytes = 0, total_bytes = 0;
				cudaMemGetInfo(&free_bytes, &total_bytes);
				throw std::runtime_error(
				  "FIDESlib: GPU memory pool could not obtain a slab of " + std::to_string(floor_bytes) +
				  " bytes for size class " + std::to_string(bytes) + " (device " + std::to_string(id) +
				  ", " + std::to_string(free_bytes >> 20) + " of " + std::to_string(total_bytes >> 20) +
				  " MiB free). The pool is keyed by exact allocation size; wholly free slabs were "
				  "already returned to the driver before this failed, so the remaining memory is held "
				  "by size classes that are still in use.");
			}

			mempool_lock[id].lock();
			uint64_t blocks = 0;
			for (uint64_t i = 0; i + (uint64_t)bytes <= request; i += (uint64_t)bytes) {
				free_limb.emplace_back(((char*)base) + i);
				++blocks;
			}
			pool_slabs[id].push_back(PoolSlab{ (char*)base, request, (size_t)bytes, (size_t)blocks });
			mempool_lock[id].unlock();
		}
		CudaCheckErrorModNoSync;

		//if (stream != nullptr) {
		s[id].record();
		CudaCheckErrorModNoSync;
		cudaStreamWaitEvent(stream, s[id].ev);
		//}

		CudaCheckErrorModNoSync;
		mempool_lock[id].lock();
		ptr = free_limb.back();
		free_limb.pop_back();
		mempool_lock[id].unlock();
		// ptr = free_limb.front();
		// free_limb.pop_front();
		// std::cout << "get " << ptr << std::endl;
		return ptr;
	}

#endif
	// std::cout << bytes << std::endl;
	if (0) {
		cudaMalloc(&ptr, bytes);
	} else if (1) {
		cudaMallocAsync(&ptr, bytes, 0);
	} else {
		if (size_to_memory[id][bytes].empty()) {
			cudaSetDevice(id);
			cudaMallocAsync(&ptr, bytes, stream);
		} else {
			mempool_lock[id].lock();
			if (size_to_memory[id][bytes].empty()) {
				mempool_lock[id].unlock();
				cudaSetDevice(id);
				cudaMallocAsync(&ptr, bytes, stream);
			} else {
				ptr = size_to_memory[id][bytes].back();
				size_to_memory[id][bytes].pop_back();
				mempool_lock[id].unlock();
			}
		}
	}

	return ptr;
}

struct pointerdata {
	void* pointer;
	int id;
	int bytes;
};

void CUDART_CB streamCallback(void* userData) {

	auto* p = reinterpret_cast<pointerdata*>(userData);

	mempool_lock[p->id].lock();
	size_to_memory[p->id][p->bytes].push_back(p->pointer);
	mempool_lock[p->id].unlock();
	delete p;
}

void GPUfree(void* ptr, int id, int bytes, cudaStream_t stream, bool cache) {

	if (bytes < 64 * 1024) {
		int next_pow2 = 1024;
		while (next_pow2 < bytes) {
			next_pow2 *= 2;
		}
		bytes = next_pow2;
		cache = true;
	}

#if MEMPOOL
	if (cache && (bytes & (bytes - 1)) == 0) {
		std::vector<void*>& free_limb = size_to_memory[id][bytes];
		if (s[id].ptr() == nullptr) {
			mempool_lock[id].lock();
			s[id].init();
			mempool_lock[id].unlock();
		}
		CudaCheckErrorModNoSync;
		// cudaDeviceSynchronize();
		//if (stream != nullptr) {
		s[id].wait(stream);
		//}
		CudaCheckErrorModNoSync;
		mempool_lock[id].lock();
		free_limb.emplace_back(ptr);
		mempool_lock[id].unlock();
		// std::cout << "free " << ptr << std::endl;
		return;
	}
#endif

	if (0) {
		cudaFree(ptr);
	} else if (1) {
		cudaFreeAsync(ptr, 0);
	} else {
		auto* p    = new pointerdata;
		p->id      = id;
		p->bytes   = bytes;
		p->pointer = ptr;
		cudaLaunchHostFunc(stream, streamCallback, p);
	}
}

int GetTargetThreads(int id) {
	return GPUprop[id].multiProcessorCount * GPUprop[id].maxThreadsPerMultiProcessor;
}

} // namespace FIDESlib