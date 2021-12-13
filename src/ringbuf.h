#ifndef _RINGBUF_H_
#define _RINGBUF_H_

typedef int64_t ringbuf_off_t;

// Single producer & consumer (same PE) ring buffer
struct ringbuf_t {
  void* ptr; // Address of ring buffer

  size_t space;
  ringbuf_off_t end;

  ringbuf_off_t write;
  ringbuf_off_t read;

  __host__ void init(size_t size);
  __host__ void fini();
  __device__ ringbuf_off_t acquire(size_t size);
  __device__ void release(size_t size);
  __device__ void print();
  __device__ void* addr(ringbuf_off_t offset) { return (void*)((char*)ptr + offset); }
};

#endif // _RINGBUF_H_
