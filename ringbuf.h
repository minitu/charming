#ifndef _RINGBUF_H_
#define _RINGBUF_H_

typedef long long ringbuf_off_t;

/* Multi-producer single-consumer (MPSC) ring buffer */

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
typedef struct ringbuf ringbuf_t;

ringbuf_t* ringbuf_malloc(size_t size);
void ringbuf_free(ringbuf_t* rbuf);

__device__ void ringbuf_init(ringbuf_t* rbuf, size_t size);

__device__ ringbuf_off_t ringbuf_acquire(ringbuf_t* rbuf, size_t size, int pe);
__device__ void ringbuf_produce(ringbuf_t* rbuf, int pe);

__device__ size_t ringbuf_consume(ringbuf_t* rbuf, size_t* offset);
__device__ void ringbuf_release(ringbuf_t* rbuf, size_t size);

/* Single producer & consumer (same PE) ring buffer */

typedef struct ringbuf single_ringbuf_t;

single_ringbuf_t* single_ringbuf_malloc(size_t size);
void single_ringbuf_free(single_ringbuf_t* rbuf);

__device__ void single_ringbuf_init(single_ringbuf_t* rbuf, size_t size);

__device__ ringbuf_off_t single_ringbuf_acquire(single_ringbuf_t* rbuf, size_t size);
__device__ void single_ringbuf_produce(single_ringbuf_t* rbuf);

__device__ size_t single_ringbuf_consume(single_ringbuf_t* rbuf, size_t* offset);
__device__ void single_ringbuf_release(single_ringbuf_t* rbuf, size_t size);


#endif // _RINGBUF_H_
