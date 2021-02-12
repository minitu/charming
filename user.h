#ifndef _USER_H_
#define _USER_H_

#include "nvcharm.h"

__device__ void hello();
__device__ int fibonacci(int n);
__device__ void register_entry_methods(int* entry_methods);
__device__ void charm_main();

struct Foo : Chare {
  int a;
  int b;

  __device__ void hello();
};

#endif
