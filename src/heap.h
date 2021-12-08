#ifndef _HEAP_H_
#define _HEAP_H_

#include <cstdint>
#include "alloy.h"

struct min_heap {
  alloy* buf;
  const int batch_size = 1;
  size_t size;
  size_t max_size;

  __device__ min_heap(alloy* buf_, size_t max_size_)
    : buf(buf_), size(0), max_size(max_size_) {}

  __device__ __forceinline__ int left(int idx) { return (idx * 2 + 1); }
  __device__ __forceinline__ int right(int idx) { return (idx * 2 + 2); }
  __device__ __forceinline__ int parent(int idx) { return ((idx - 1) / 2); }
  __device__ __forceinline__ alloy* addr(int idx) { return (buf + batch_size * idx); }

  // TODO: Use a generalized heap for vector operations
  __device__ int push(const alloy& key);
  __device__ alloy pop();
  __device__ void print();
};

#endif // _HEAP_H_
