#ifndef _RINGBUF_H_
#define _RINGBUF_H_

typedef int64_t ringbuf_off_t;

// Single producer & consumer (same PE) ring buffer
struct ringbuf_t {
  size_t space;

  // Point to starting address of ring buffer
  void* ptr;

  // Atomically updated by the producer
  ringbuf_off_t next;
  ringbuf_off_t end;

  // Updated by the consumer
  ringbuf_off_t written;

  __host__ void init(size_t size);
  __host__ void fini();
  __device__ ringbuf_off_t acquire(size_t size);
  __device__ size_t consume(size_t* offset);
  __device__ void release(size_t size);
  __device__ void* addr(ringbuf_off_t offset) { return (void*)((char*)ptr + offset); }
};

#endif // _RINGBUF_H_
