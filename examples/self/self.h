#ifndef _SELF_H_
#define _SELF_H_

#include <cuda/std/chrono>
#include <charming.h>

struct Comm : charm::chare {
  int init_cnt;
  int index;
  int self;
  size_t min_size;
  size_t max_size;
  size_t cur_size;
  int n_iters;
  int warmup;
  int iter;
#ifdef USER_MSG
  charm::message msg;
#else
  char* data;
#endif

  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> start_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> end_tp;

  __device__ Comm() {}
  __device__ void init(void* arg);
  __device__ void run(void* arg);
};

// Entry methods
__device__ void entry_init(Comm& c, void* arg) { c.init(arg); }
__device__ void entry_run(Comm& c, void* arg) { c.run(arg); }

#endif
