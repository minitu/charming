#include "ringbuf.h"
#include <nvshmem.h>
#include <cassert>

#define SPINLOCK_BACKOFF_MIN 4
#define SPINLOCK_BACKOFF_MAX 128
// TODO: Not really exponential backoff
#define SPINLOCK_BACKOFF(count)         \
  do {                                  \
    if ((count) < SPINLOCK_BACKOFF_MAX) \
      (count) += (count);               \
  } while (0);

#define RBUF_OFF_MASK (0x00000000ffffffffUL)
#define WRAP_LOCK_BIT (0x8000000000000000UL)
#define RBUF_OFF_MAX  (UINT64_MAX & ~WRAP_LOCK_BIT)

#define WRAP_COUNTER  (0x7fffffff00000000UL)
#define WRAP_INCR(x)  (((x) + 0x100000000UL) & WRAP_COUNTER)

/* Single producer & consumer (same PE) ring buffer */

// NVSHMEM allocation: ringbuf metadata + actual buffer

spsc_ringbuf_t* spsc_ringbuf_malloc(size_t size) {
  size_t alloc_size = sizeof(spsc_ringbuf_t) + size;
  spsc_ringbuf_t* ringbuf_all = (spsc_ringbuf_t*)nvshmem_malloc(alloc_size);
  assert(ringbuf_all);
  return ringbuf_all;
}

void spsc_ringbuf_free(spsc_ringbuf_t* rbuf) {
  nvshmem_free(rbuf);
}

__device__ void spsc_ringbuf_init(spsc_ringbuf_t* rbuf, size_t size) {
  rbuf->space = size;
  rbuf->ptr = (char*)rbuf + sizeof(spsc_ringbuf_t);
  rbuf->next = 0;
  rbuf->end = RBUF_OFF_MAX;
  rbuf->written = 0;
}

__device__ ringbuf_off_t spsc_ringbuf_acquire(spsc_ringbuf_t* rbuf, size_t size) {
  ringbuf_off_t seen, next, target, written;

  seen = rbuf->next;
  next = seen & RBUF_OFF_MASK;
  assert(next < rbuf->space);

  target = next + size;
  written = rbuf->written;
  // TODO: Can't target be equal to written?
  if (next < written && target >= written) {
    return -1;
  }

  if (target >= rbuf->space) {
    // Wrap-around
    const bool exceed = target > rbuf->space;
    target = exceed ? (WRAP_LOCK_BIT | size) : 0;
    if ((target & RBUF_OFF_MASK) >= written) {
      return -1;
    }
    target |= WRAP_INCR(seen & WRAP_COUNTER);
  } else {
    target |= seen & WRAP_COUNTER;
  }

  rbuf->next = target;

  if (target & WRAP_LOCK_BIT) {
    assert(rbuf->written <= next);
    assert(rbuf->end == RBUF_OFF_MAX);

    rbuf->end = next;
    next = 0;

    rbuf->next = target & ~WRAP_LOCK_BIT;
  }
  assert((target & RBUF_OFF_MASK) <= rbuf->space);
  return next;
}

__device__ void spsc_ringbuf_produce(spsc_ringbuf_t* rbuf) {
}

__device__ size_t spsc_ringbuf_consume(spsc_ringbuf_t* rbuf, size_t* offset) {
  ringbuf_off_t written = rbuf->written, next, ready;
  size_t towrite;
retry:
  next = rbuf->next & RBUF_OFF_MASK;
  if (written == next) {
    // Producers did not advance
    return 0;
  }

  ready = RBUF_OFF_MAX;

  // Determine whether wrap-around occurred and deduct safe 'ready' offset
  if (next < written) {
    const ringbuf_off_t end = llmin(rbuf->space, rbuf->end);

    if (ready == RBUF_OFF_MAX && written == end) {
      // TODO: Is this necessary?
      if (rbuf->end != RBUF_OFF_MAX) {
        rbuf->end = RBUF_OFF_MAX;
      }

      rbuf->written = 0;
      goto retry;
    }

    assert(ready > next);
    ready = llmin(ready, end);
    assert(ready >= written);
  } else {
    ready = llmin(ready, next);
  }
  towrite = ready - written;
  *offset = written;

  assert(ready >= written);
  assert(towrite <= rbuf->space);
  return towrite;
}

__device__ void spsc_ringbuf_release(spsc_ringbuf_t* rbuf, size_t size) {
  const size_t written = rbuf->written + size;

  assert(rbuf->written <= rbuf->space);
  assert(rbuf->written <= rbuf->end);
  assert(written <= rbuf->space);

  rbuf->written = (written == rbuf->space) ? 0 : written;
}
