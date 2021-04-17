#ifndef _JACOBI1D_H_
#define _JACOBI1D_H_

#include <charming.h>

struct Block : charm::chare {
  __device__ Block() {}
  __device__ void foo(void* arg);
};

#endif
