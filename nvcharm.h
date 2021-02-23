#ifndef _NVCHARM_H_
#define _NVCHARM_H_

#include <nvfunctional>

struct DeviceCtx {
  int n_sms;
};

struct Proxy {
  int n_chares;
  int *mapping; // Chare -> SM mapping

  __device__ Proxy() : n_chares(-1), mapping(nullptr) {}
};

struct Chare {
  Proxy proxy;

  __device__ void create(int n_chares);
  __device__ void invoke(int ep, int idx);
};

struct EntryMethod {
  int idx;
  __device__ virtual void call() const = 0;
};

template <typename T>
struct EntryMethodImpl : EntryMethod {
  nvstd::function<T> fn;

  __device__ EntryMethodImpl(nvstd::function<T> fn_) : fn(fn_) {}
  __device__ virtual void call() const { fn(); }
};

// User functions required by the runtime
__device__ void register_entry_methods(EntryMethod** entry_methods);
__device__ void charm_main();

#endif
