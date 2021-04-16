#ifndef _HELLO_H_
#define _HELLO_H_

#include <charming.h>

struct Hello : charm::chare {
  __device__ Hello() {}
  __device__ void greet(void* arg);
};

#endif
