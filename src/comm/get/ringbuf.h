#ifndef _RINGBUF_H_
#define _RINGBUF_H_

#include <stdint.h>

typedef int64_t ringbuf_off_t;
typedef int64_t compbuf_off_t;

// Single producer & consumer (same PE) ring buffer
struct ringbuf_t {
  void* base; // Base address of ring buffer
  ringbuf_off_t start_offset; // Starting offset of this SM

  size_t space;
  ringbuf_off_t end;

  ringbuf_off_t write;
  ringbuf_off_t read;

  __host__ void init(void* ptr, size_t size, int my_sm);
  __device__ ringbuf_off_t acquire(size_t size);
  __device__ void release(size_t size);
  __device__ void print();
  __device__ void* addr(ringbuf_off_t offset) {
    return (void*)((char*)base + offset);
  }
};

// Ring buffer that contains composites
struct compbuf_t {
  uint64_t* ptr; // Address of ring buffer

  size_t max;
  size_t count;
  compbuf_off_t write;
  compbuf_off_t read;

  __host__ void init(size_t max_);
  __host__ void fini();
  __device__ compbuf_off_t acquire();
  __device__ void release();
  __device__ void print();
  __device__ uint64_t* addr(compbuf_off_t offset) {
    return ptr + offset;
  }
};

#endif // _RINGBUF_H_
