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

__host__ void ringbuf_t::init(size_t size) {
  space = size;
  next = 0;
  end = RBUF_OFF_MAX;
  written = 0;

  // Allocate NVSHMEM memory
  ptr = nvshmem_malloc(space);
  assert(ptr);
}

__host__ void ringbuf_t::fini() {
  nvshmem_free(ptr);
}

__device__ ringbuf_off_t ringbuf_t::acquire(size_t size) {
  ringbuf_off_t seen, target;
  ringbuf_off_t l_next, l_written;

  seen = next;
  l_next = seen & RBUF_OFF_MASK;
  assert(l_next < space);

  target = l_next + size;
  l_written = written;
  // TODO: Can't target be equal to written?
  if (l_next < l_written && target >= l_written) {
    return -1;
  }

  if (target >= space) {
    // Wrap-around
    const bool exceed = target > space;
    target = exceed ? (WRAP_LOCK_BIT | size) : 0;
    if ((target & RBUF_OFF_MASK) >= l_written) {
      return -1;
    }
    target |= WRAP_INCR(seen & WRAP_COUNTER);
  } else {
    target |= seen & WRAP_COUNTER;
  }

  next = target;

  if (target & WRAP_LOCK_BIT) {
    assert(written <= l_next);
    assert(end == RBUF_OFF_MAX);

    end = l_next;
    l_next = 0;

    next = target & ~WRAP_LOCK_BIT;
  }
  assert((target & RBUF_OFF_MASK) <= space);
  return l_next;
}

__device__ size_t ringbuf_t::consume(size_t* offset) {
  ringbuf_off_t l_written = written, l_next, ready;
  size_t towrite;
retry:
  l_next = next & RBUF_OFF_MASK;
  if (l_written == l_next) {
    // Producers did not advance
    return 0;
  }

  ready = RBUF_OFF_MAX;

  // Determine whether wrap-around occurred and deduct safe 'ready' offset
  if (l_next < l_written) {
    const ringbuf_off_t l_end = llmin(space, end);

    if (ready == RBUF_OFF_MAX && l_written == l_end) {
      // TODO: Is this necessary?
      if (end != RBUF_OFF_MAX) {
        end = RBUF_OFF_MAX;
      }

      written = 0;
      goto retry;
    }

    assert(ready > l_next);
    ready = llmin(ready, l_end);
    assert(ready >= l_written);
  } else {
    ready = llmin(ready, l_next);
  }
  towrite = ready - l_written;
  *offset = l_written;

#ifdef DEBUG
  assert(ready >= l_written);
  assert(towrite <= space);
#endif

  return towrite;
}

__device__ void ringbuf_t::release(size_t size) {
  const size_t new_written = written + size;

#ifdef DEBUG
  assert(written <= space);
  assert(written <= end);
  assert(new_written <= space);
#endif

  written = (new_written == space) ? 0 : new_written;
}
