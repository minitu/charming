#ifndef _REFNUM_H_
#define _REFNUM_H_

#include <cuda/std/chrono>
#include <charming.h>

// Chare object
struct Comm : charm::chare {
  int index;
  int peer;
  char* data;

  __device__ Comm() {}
  __device__ void send(void* arg);
  __device__ void recv(void* arg);
};

// Entry methods
__device__ void entry_send(Comm& c, void* arg) { c.send(arg); }
__device__ void entry_recv(Comm& c, void* arg) { c.recv(arg); }

#endif
