#ifndef _HELLO_H_
#define _HELLO_H_

#include <charming.h>

struct Hello {
  __device__ Hello() {}
  __device__ void greet(void* arg);

  __device__ size_t pack_size();
  __device__ void pack(void* ptr);
  __device__ void unpack(void* ptr);
};

#endif
