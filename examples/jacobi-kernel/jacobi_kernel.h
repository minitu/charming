#ifndef _JACOBI_KERNEL_H_
#define _JACOBI_KERNEL_H_

#include <cuda.h>

#define IDX_2D(i,j) ((width+2)*(j)+(i))
#define IDX_3D(i,j,k) ((width+2)*(height+2)*(k)+(width+2)*(j)+(i))

typedef double DataType;

struct Block {
  int width;
  int height;
  int depth;
  int n_iters;
  int warmup_iters;

  int i;
  int j;
  int k;
  int dims;

  DataType* temp;
  DataType* new_temp;

  cudaStream_t stream;

  Block(int width_, int height_, int depth_, int n_iters_, int warmup_iters_);
  ~Block();
  void run();
};

#endif // _JACOBI_KERNEL_H_
