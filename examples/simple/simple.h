#ifndef _SIMPLE_H_
#define _SIMPLE_H_

#include <charming.h>

struct Foo : charm::chare {
  int my_int;

  __device__ Foo() {}
  __device__ Foo(int my_int_) : my_int(my_int_) {}
  __device__ void hello(void* arg);
  __device__ void morning(void* arg);

  __device__ size_t pack_size();
  __device__ void pack(void* ptr);
  __device__ void unpack(void* ptr);
};

struct Bar : charm::chare {
  char my_char;

  __device__ Bar() {}
  __device__ Bar(char my_char_) : my_char(my_char_) {}
  __device__ void hammer(void* arg);

  __device__ size_t pack_size();
  __device__ void pack(void* ptr);
  __device__ void unpack(void* ptr);
};

//template struct charm::chare<Foo>;
//template struct charm::chare<Bar>;

#endif
