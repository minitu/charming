#ifndef _SIMPLE_H_
#define _SIMPLE_H_

#include <charming.h>

struct Foo {
  int i;

  __device__ Foo() {}
  __device__ Foo(int i_) : i(i_) {}
  __device__ void hello(void* arg);
  __device__ void morning(void* arg);

  __device__ size_t pack_size();
  __device__ void pack(void* ptr);
  __device__ void unpack(void* ptr);
};

struct Bar {
  char ch;

  __device__ Bar() {}
  __device__ Bar(char ch_) : ch(ch_) {}
  __device__ void hammer(void* arg);

  __device__ size_t pack_size();
  __device__ void pack(void* ptr);
  __device__ void unpack(void* ptr);
};

//template struct charm::chare<Foo>;
//template struct charm::chare<Bar>;

#endif
