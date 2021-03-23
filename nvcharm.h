#ifndef _NVCHARM_H_
#define _NVCHARM_H_

#include <nvfunctional>

#define cuda_check_error() {                                                         \
  cudaError_t e = cudaGetLastError();                                                \
  if (cudaSuccess != e) {                                                            \
    printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(-1);                                                                        \
  }                                                                                  \
}

struct EntryMethod {
  int idx; // FIXME: Needed?

  __device__ EntryMethod(int idx_) : idx(idx_) {}
  //__device__ virtual void call() const = 0;
};

template <typename T>
struct EntryMethodImpl : EntryMethod {
  nvstd::function<T> fn;

  __device__ EntryMethodImpl(int idx, nvstd::function<T> fn_)
    : EntryMethod(idx), fn(fn_) {}
  //__device__ virtual void call() const { fn(); }
  __device__ void call(T chare) const { fn(chare); }
};

struct ChareType {
  int id; // FIXME: Needed?

  __device__ ChareType(int id_);
};

template <typename T>
struct Chare : ChareType {
  T* obj;
  EntryMethod** entry_methods;

  __device__ Chare(int id_);
  __device__ void invoke(int ep, int idx);
};

// User functions required by the runtime
__device__ void register_chare_types(ChareType** chare_types);
__device__ void charm_main();

#endif
