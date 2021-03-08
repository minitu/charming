#ifndef _RINGBUF_H_
#define _RINGBUF_H_

typedef struct ringbuf ringbuf_t;
typedef long long ringbuf_off_t;

ringbuf_t* ringbuf_malloc(size_t size);
void ringbuf_free(ringbuf_t* rbuf);

__device__ void ringbuf_init(ringbuf_t* rbuf, size_t size);

__device__ ringbuf_off_t ringbuf_acquire(ringbuf_t* rbuf, size_t size, int pe);
__device__ void ringbuf_produce(ringbuf_t* rbuf, int pe);

__device__ size_t ringbuf_consume(ringbuf_t* rbuf, size_t* offset);
__device__ void ringbuf_release(ringbuf_t* rbuf, size_t size);

#endif // _RINGBUF_H_
