#ifndef _PINGPONG_H_
#define _PINGPONG_H_

#include <cuda/std/chrono>
#include <charming.h>

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
  char* data;

  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> start_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> end_tp;

  __device__ Comm() {}
  __device__ void init(void* arg);
  __device__ void init_done(void* arg);
  __device__ void send();
  __device__ void recv(void* arg);
};

#endif
