#ifndef _JACOBI1D_H_
#define _JACOBI1D_H_

#include <charming.h>

#define ALIGN_SIZE 16
#define GHOST_SIZE 2

typedef double DataType;

struct alignas(ALIGN_SIZE) Ghost {
  int dir;
  DataType data[GHOST_SIZE];

  __device__ Ghost(int dir_) : dir(dir_) {}
};

struct Block : charm::chare {
  int index;
  int iter;
  int block_width;
  int data_size;
  DataType* temperature;
  DataType* new_temperature;
  int left_index;
  int right_index;
  size_t ghost_size;
  Ghost* left_ghost;
  Ghost* right_ghost;
  int neighbor_count;
  int recv_count;

  __device__ Block() {}
  __device__ void init(void* arg);
  __device__ void send_ghosts();
  __device__ void recv_ghosts(void* arg);
  __device__ void update();
};

#endif
