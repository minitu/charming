#include "ringbuf.h"
#include <nvshmem.h>
#include <cassert>

#define SPINLOCK_BACKOFF_MIN 4
#define SPINLOCK_BACKOFF_MAX 128
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

/* Multi-producer single-consumer (MPSC) ring buffer */

// NVSHMEM allocation: ringbuf metadata + seen_off * n_pes + actual buffer

__device__ inline ringbuf_off_t* seen_off_addr(ringbuf_t* rbuf, int pe) {
  return (ringbuf_off_t*)((char*)rbuf + sizeof(ringbuf_t) + sizeof(ringbuf_off_t) * pe);
}

ringbuf_t* ringbuf_malloc(size_t size) {
  size_t alloc_size = sizeof(ringbuf_t) + sizeof(ringbuf_off_t) * nvshmem_n_pes() + size;
  ringbuf_t* ringbuf_all = (ringbuf_t*)nvshmem_malloc(alloc_size);
  assert(ringbuf_all);
  return ringbuf_all;
}

void ringbuf_free(ringbuf_t* rbuf) {
  nvshmem_free(rbuf);
}

__device__ void ringbuf_init(ringbuf_t* rbuf, size_t size) {
  int n_pes = nvshmem_n_pes();
  rbuf->space = size;
  rbuf->ptr = seen_off_addr(rbuf, n_pes);
  rbuf->next = 0;
  rbuf->end = RBUF_OFF_MAX;
  rbuf->written = 0;
  for (int pe = 0; pe < n_pes; pe++) {
    *seen_off_addr(rbuf, pe) = RBUF_OFF_MAX;
  }
}

// Atomically fetch 'next' until it doesn't have WRAP_LOCK_BIT set
inline __device__ ringbuf_off_t stable_next_off(ringbuf_t* rbuf, int pe) {
  unsigned count = SPINLOCK_BACKOFF_MIN;
  ringbuf_off_t next;
retry:
  next = nvshmem_longlong_atomic_fetch(&rbuf->next, pe);
  if (next & WRAP_LOCK_BIT) {
    SPINLOCK_BACKOFF(count);
    goto retry;
  }
  assert((next & RBUF_OFF_MASK) < rbuf->space);
  return next;
}

// Atomically fetch 'seen_off' until it doesn't have WRAP_LOCK_BIT set
inline __device__ ringbuf_off_t stable_seen_off(ringbuf_t* rbuf, int pe) {
  int my_pe = nvshmem_my_pe();
  unsigned count = SPINLOCK_BACKOFF_MIN;
  ringbuf_off_t seen_off;
retry:
  seen_off = nvshmem_longlong_atomic_fetch(seen_off_addr(rbuf, pe), my_pe);
  if (seen_off & WRAP_LOCK_BIT) {
    SPINLOCK_BACKOFF(count);
    goto retry;
  }
  return seen_off;
}

__device__ ringbuf_off_t ringbuf_acquire(ringbuf_t* rbuf, size_t size, int pe) {
  int my_pe = nvshmem_my_pe();
  ringbuf_off_t seen, next, target;

  assert(size > 0 && size <= rbuf->space);
  /* Removed assertion for performance
  ringbuf_off_t remote_seen_off = nvshmem_longlong_atomic_fetch(seen_off_addr(rbuf, my_pe), pe);
  assert(remote_seen_off == RBUF_OFF_MAX);
  */

  do {
    ringbuf_off_t written;

    seen = stable_next_off(rbuf, pe);
    next = seen & RBUF_OFF_MASK;
    assert(next < rbuf->space);
    nvshmem_longlong_atomic_set(seen_off_addr(rbuf, my_pe), next | WRAP_LOCK_BIT, pe);

    target = next + size;
    written = nvshmem_longlong_atomic_fetch(&rbuf->written, pe);
    if (next < written && target >= written) {
      // There isn't enough space, producer must wait
      nvshmem_longlong_atomic_set(seen_off_addr(rbuf, my_pe), RBUF_OFF_MAX, pe);
      return -1;
    }

    if (target >= rbuf->space) {
      // Wrap-around
      const bool exceed = target > rbuf->space;
      target = exceed ? (WRAP_LOCK_BIT | size) : 0;
      if ((target & RBUF_OFF_MASK) >= written) {
        nvshmem_longlong_atomic_set(seen_off_addr(rbuf, my_pe), RBUF_OFF_MAX, pe);
        return -1;
      }
      target |= WRAP_INCR(seen & WRAP_COUNTER);
    } else {
      target |= seen & WRAP_COUNTER;
    }
  } while (seen != nvshmem_longlong_atomic_compare_swap(&rbuf->next, seen, target, pe));

  nvshmem_ulonglong_atomic_fetch_and((unsigned long long*)seen_off_addr(rbuf, my_pe), ~WRAP_LOCK_BIT, pe);

  if (target & WRAP_LOCK_BIT) {
    // TODO: Remove assertions for performance?
    ringbuf_off_t written = nvshmem_longlong_atomic_fetch(&rbuf->written, pe);
    ringbuf_off_t end = nvshmem_longlong_atomic_fetch(&rbuf->end, pe);
    assert(written <= next);
    assert(end == RBUF_OFF_MAX);

    nvshmem_longlong_atomic_set(&rbuf->end, next, pe);
    next = 0;

    nvshmem_longlong_atomic_set(&rbuf->next, (target & ~WRAP_LOCK_BIT), pe);
  }
  assert((target & RBUF_OFF_MASK) <= rbuf->space);
  return next;
}

__device__ void ringbuf_produce(ringbuf_t* rbuf, int pe) {
  int my_pe = nvshmem_my_pe();
  /* Removed assertion for performance
  ringbuf_off_t remote_seen_off = nvshmem_longlong_atomic_fetch(seen_off_addr(rbuf, my_pe), pe);
  assert(remote_seen_off != RBUF_OFF_MAX);
  */
  nvshmem_longlong_atomic_set(seen_off_addr(rbuf, my_pe), RBUF_OFF_MAX, pe);
}

__device__ size_t ringbuf_consume(ringbuf_t* rbuf, size_t* offset) {
  int my_pe = nvshmem_my_pe();
  ringbuf_off_t written = rbuf->written, next, ready;
  size_t towrite;
retry:
  next = stable_next_off(rbuf, my_pe) & RBUF_OFF_MASK;
  if (written == next) {
    // Producers did not advance
    return 0;
  }

  // Check all producers
  ready = RBUF_OFF_MAX;
  for (int i = 0; i < nvshmem_n_pes(); i++) {
    ringbuf_off_t seen_off = stable_seen_off(rbuf, i);
    if (seen_off >= written) {
      ready = llmin(seen_off, ready);
    }
    assert(ready >= written);
  }

  // Determine whether wrap-around occurred and deduct safe 'ready' offset
  if (next < written) {
    const ringbuf_off_t end = llmin(rbuf->space, rbuf->end);

    if (ready == RBUF_OFF_MAX && written == end) {
      // TODO: OK to not use atomics?
      if (rbuf->end != RBUF_OFF_MAX) {
        rbuf->end = RBUF_OFF_MAX;
      }

      written = 0;
      nvshmem_longlong_atomic_set(&rbuf->written, written, my_pe);
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

__device__ void ringbuf_release(ringbuf_t* rbuf, size_t size) {
  const size_t written = rbuf->written + size;

  assert(rbuf->written <= rbuf->space);
  assert(rbuf->written <= rbuf->end);
  assert(written <= rbuf->space);

  rbuf->written = (written == rbuf->space) ? 0 : written;
}

/* Single producer & consumer (same PE) ring buffer */

// NVSHMEM allocation: ringbuf metadata + actual buffer

single_ringbuf_t* single_ringbuf_malloc(size_t size) {
  size_t alloc_size = sizeof(ringbuf_t) + size;
  single_ringbuf_t* ringbuf_all = (single_ringbuf_t*)nvshmem_malloc(alloc_size);
  assert(ringbuf_all);
  return ringbuf_all;
}

void single_ringbuf_free(single_ringbuf_t* rbuf) {
  nvshmem_free(rbuf);
}

__device__ void single_ringbuf_init(single_ringbuf_t* rbuf, size_t size) {
  rbuf->space = size;
  rbuf->ptr = (char*)rbuf + sizeof(ringbuf_t);
  rbuf->next = 0;
  rbuf->end = RBUF_OFF_MAX;
  rbuf->written = 0;
}

__device__ ringbuf_off_t single_ringbuf_acquire(single_ringbuf_t* rbuf, size_t size) {
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

__device__ void single_ringbuf_produce(single_ringbuf_t* rbuf) {
}

__device__ size_t single_ringbuf_consume(single_ringbuf_t* rbuf, size_t* offset) {
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

__device__ void single_ringbuf_release(single_ringbuf_t* rbuf, size_t size) {
  const size_t written = rbuf->written + size;

  assert(rbuf->written <= rbuf->space);
  assert(rbuf->written <= rbuf->end);
  assert(written <= rbuf->space);

  rbuf->written = (written == rbuf->space) ? 0 : written;
}
