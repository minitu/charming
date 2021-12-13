#include "ringbuf.h"
#include <nvshmem.h>
#include <cassert>

__host__ void ringbuf_t::init(size_t size) {
  space = size;
  write = 0;
  end = UINT64_MAX;
  read = 0;

  // Allocate NVSHMEM memory
  ptr = nvshmem_malloc(space);
  assert(ptr);
}

__host__ void ringbuf_t::fini() {
  nvshmem_free(ptr);
}

__device__ ringbuf_off_t ringbuf_t::acquire(size_t size) {
#ifdef DEBUG
  assert(write < space);
#endif

  // Compute potential new write offset
  ringbuf_off_t new_write = write + size;
  if (write < read && new_write >= read) {
    // Next should not catch up to read
    return -1;
  }

  ringbuf_off_t write_ret = write;
  if (new_write >= space) {
    // Need to Wrap-around
    const bool exceed = new_write > space;
    if (exceed) {
      // Early wrap-around
#ifdef DEBUG
      assert(read <= write);
      assert(end == UINT64_MAX);
#endif
      end = write;
      new_write = size;
      write_ret = 0;
    } else {
      // Exact wrap-around
      new_write = 0;
    }
    if (new_write >= read) {
      return -1;
    }
  }

  // Update write value
#ifdef DEBUG
  assert(new_write <= space);
#endif
  write = new_write;

  return write_ret;
}

__device__ void ringbuf_t::release(size_t size) {
#ifdef DEBUG
  assert((read <= write && (read + size <= write))
      || (read > write && (read + size <= end)));
#endif
  read += size;
  if (read == end) {
    end = UINT64_MAX;
    read = 0;
  }
}

__device__ void ringbuf_t::print() {
  printf("[ringbuf_t] ptr: %p, space: %llu, end: %llu, write: %llu, read: %llu\n",
      ptr, space, end, write, read);
}
