#ifndef _PINGPONG_H_
#define _PINGPONG_H_

#include <cuda/std/chrono>
#include <charming.h>

// Chare object
struct Comm : charm::chare {
  int init_cnt;
  int index;
  int peer;
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
#ifndef SM_LEVEL
  bool end;
#endif

  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> start_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> end_tp;

#ifdef MEASURE_INVOKE
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> invoke_start_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> invoke_end_tp;
  double invoke_time;
#endif

  __device__ Comm() {}
  __device__ void init(void* arg);
  __device__ void init_done(void* arg);
  __device__ void send();
  __device__ void recv(void* arg);
};

// Entry methods
__device__ void entry_init(Comm& c, void* arg) { c.init(arg); }
__device__ void entry_init_done(Comm& c, void* arg) { c.init_done(arg); }
__device__ void entry_recv(Comm& c, void* arg) { c.recv(arg); }

#endif
