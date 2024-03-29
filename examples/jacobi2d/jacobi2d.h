#ifndef _JACOBI2D_H_
#define _JACOBI2D_H_

#include <charming.h>
#include <cuda/std/chrono>

#define ALIGN_SIZE 16
#define N_NEIGHBORS 4

typedef double DataType;

struct Block : charm::chare {
  int row;
  int col;
  int neighbor_index[N_NEIGHBORS];
  int neighbor_count;
  int recv_count;
  int end_count;

  int block_width;
  int block_height;
  unsigned long long block_size;
  int n_iters;
  int iter;

  DataType* temperature;
  DataType* new_temperature;
  DataType* boundaries[N_NEIGHBORS];
  size_t ghost_sizes[N_NEIGHBORS];

  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> start_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> end_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> send1_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> send2_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> send_end_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> recv_end_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> update_end_tp;
  cuda::std::chrono::duration<double> temp_diff;
  double send1_time;
  double send2_time;
  double recv_time;
  double update_time;

  __device__ Block() {}
  __device__ void init(void* arg);
  __device__ void send_boundaries();
  __device__ void recv_ghosts(void* arg);
  __device__ void update();
  __device__ void end(void* arg);
};

#endif
