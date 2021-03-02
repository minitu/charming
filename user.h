#ifndef _USER_H_
#define _USER_H_

#include "nvcharm.h"

struct Foo {
  int a;

  __device__ Foo() {}
  __device__ Foo(int a_) : a(a_) {}
  __device__ void hello();
  __device__ void morning();
};

template struct Chare<Foo>;

#endif
