#ifndef _RINGBUF_H_
#define _RINGBUF_H_

typedef long long ringbuf_off_t;

struct ringbuf {
  size_t space;

  // Point to starting address of ring buffer
  void* ptr;

  // Atomically updated by the producer
  ringbuf_off_t next;
  ringbuf_off_t end;

  // Updated by the consumer
  ringbuf_off_t written;

  __device__ void* addr(ringbuf_off_t offset) { return (void*)((char*)ptr + offset); }
};

/* Single producer & consumer (same PE) ring buffer */

typedef struct ringbuf spsc_ringbuf_t;

spsc_ringbuf_t* spsc_ringbuf_malloc(size_t size);
void spsc_ringbuf_free(spsc_ringbuf_t* rbuf);

__device__ void spsc_ringbuf_init(spsc_ringbuf_t* rbuf, size_t size);

__device__ ringbuf_off_t spsc_ringbuf_acquire(spsc_ringbuf_t* rbuf, size_t size);
__device__ void spsc_ringbuf_produce(spsc_ringbuf_t* rbuf);

__device__ size_t spsc_ringbuf_consume(spsc_ringbuf_t* rbuf, size_t* offset);
__device__ void spsc_ringbuf_release(spsc_ringbuf_t* rbuf, size_t size);


#endif // _RINGBUF_H_
