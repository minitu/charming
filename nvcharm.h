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
  __device__ virtual void call(void* chare) const = 0;
};

template <typename C, typename T>
struct EntryMethodImpl : EntryMethod {
  nvstd::function<T> fn;

  __device__ EntryMethodImpl(int idx, nvstd::function<T> fn_)
    : EntryMethod(idx), fn(fn_) {}
  __device__ virtual void call(void* chare) const { fn(*(C*)chare); }
};

struct ChareType {
  int id; // FIXME: Needed?

  __device__ ChareType(int id_) : id(id_) {}
  __device__ virtual void alloc() = 0;
  __device__ virtual void unpack(void* ptr) = 0;
  __device__ virtual void call(int ep) = 0;
};

template <typename C>
struct Chare : ChareType {
  C* obj;
  EntryMethod** entry_methods;

  __device__ Chare(int id_) : ChareType(id_), obj(nullptr), entry_methods(nullptr) {}
  __device__ void create(C& obj_);
  __device__ void invoke(int idx, int ep);
  __device__ void alloc(C& obj_) { obj = new C(obj_); }
  __device__ virtual void alloc() { obj = new C; }
  __device__ virtual void unpack(void* ptr) { obj->unpack(ptr); }
  __device__ virtual void call(int ep) { entry_methods[ep]->call(obj); }
};

// User functions required by the runtime
__device__ void register_chare_types(ChareType** chare_types);
__device__ void charm_main(ChareType** chare_types);

// Runtime functions that can be called by the user
__device__ void ckExit();

#endif
