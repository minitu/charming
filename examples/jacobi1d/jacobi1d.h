#ifndef _JACOBI1D_H_
#define _JACOBI1D_H_

#include <charming.h>

#define GHOST_SIZE 2

typedef double DataType;

struct Block : charm::chare {
  DataType* temperature;
  DataType* new_temperature;

  __device__ Block() {}
  __device__ void init(void* arg);
  __device__ void send_halo(void* arg);
};

#endif
