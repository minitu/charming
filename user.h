#ifndef _USER_H_
#define _USER_H_

#include "nvcharm.h"

struct Foo {
  int i;

  __device__ Foo() {}
  __device__ Foo(int i_) : i(i_) {}
  __device__ void hello();
  __device__ void morning();

  __device__ size_t pack_size();
  __device__ void pack(void* ptr);
  __device__ void unpack(void* ptr);
};

struct Bar {
  char ch;

  __device__ Bar() {}
  __device__ Bar(char ch_) : ch(ch_) {}
  __device__ void hammer();

  __device__ size_t pack_size();
  __device__ void pack(void* ptr);
  __device__ void unpack(void* ptr);
};

template struct Chare<Foo>;
template struct Chare<Bar>;

#endif
