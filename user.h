#ifndef _USER_H_
#define _USER_H_

#include "nvcharm.h"

__device__ void hello();
__device__ void morning();

struct Foo : ChareArray {
  int a;

  __device__ Foo(int a_, int n_chares) : a(a_), ChareArray(n_chares) {}
  __device__ void hello();
  __device__ void morning();
};

#endif
