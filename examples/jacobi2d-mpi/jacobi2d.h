#ifndef _JACOBI2D_H_
#define _JACOBI2D_H_

#include <mpi.h>
#include <cuda/std/chrono>

#define ALIGN_SIZE 16
#define N_NEIGHBORS 4

typedef double DataType;

struct Block {
  int index;
  int row;
  int col;
  int neighbor_index[N_NEIGHBORS];
  int neighbor_count;

  int block_width;
  int block_height;
  unsigned long long block_size;
  int n_iters;
  int iter;

  DataType* temperature;
  DataType* new_temperature;
  DataType* boundaries[N_NEIGHBORS];
  DataType* ghosts[N_NEIGHBORS];
  size_t ghost_sizes[N_NEIGHBORS];

  MPI_Request requests[N_NEIGHBORS * 2];
  MPI_Status statuses[N_NEIGHBORS * 2];

  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> start_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> end_tp;
  double max_time;

  Block() {}
  void init(void* arg);
  void send_boundaries();
  void recv_ghosts();
  void update();
};

#endif
