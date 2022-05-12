#include "ringbuf.h"
#include <nvshmem.h>
#include <cassert>

__host__ void ringbuf_t::init(void* ptr, size_t size, int my_sm) {
  base = ptr;
  start_offset = size * my_sm;
  space = size;
  watermark = UINT64_MAX;
  write = 0;
  read = 0;
}

__device__ bool ringbuf_t::acquire(size_t size, size_t& ret_offset) {
  // Compute potential new write offset
  size_t old_write = write;
  size_t new_write = write + size;

  if (new_write <= space) {
    // There is enough space
    // Check if new write catches up to read
    if (write < read && new_write >= read) {
      return false;
    }

    write = new_write;
    ret_offset = start_offset + old_write;
  } else {
    // Not enough space, need to wrap around
    // Check if new write catches up to read
    new_write = size;
    if (new_write >= read) {
      return false;
    }

    // If read and write are at the same position when wrapping around,
    // move read to beginning instead of setting watermark
    if (write == read) {
      read = 0;
      watermark = UINT64_MAX;
    } else {
      watermark = write;
    }
    write = new_write;
    ret_offset = start_offset;
  }

  return true;
}

__device__ bool ringbuf_t::release(size_t size) {
  if ((read <= write && (read + size) > write)
      || (read > write && (read + size) > watermark)) {
    return false;
  }

  read += size;
  if (read == watermark) {
    read = 0;
    watermark = UINT64_MAX;
  }

  return true;
}

__device__ void ringbuf_t::print() {
  printf("[ringbuf_t] base: %p, start_offset: %llu, space: %llu, watermark: %llu, "
      "write: %lld, read: %lld\n",
      base, start_offset, space, watermark, write, read);
}

__host__ void compbuf_t::init(size_t max_) {
  max = max_;
  count = 0;
  write = 0;
  read = 0;

  // Allocate memory
  cudaMalloc(&ptr, sizeof(uint64_t) * max);
  assert(ptr);
}

__host__ void compbuf_t::fini() {
  cudaFree(ptr);
}

__device__ compbuf_off_t compbuf_t::acquire() {
  // Check if there is space
  if (count >= max) return -1;

  compbuf_off_t write_ret = write++;
  if (write == max) {
    // Wrap around
    write = 0;
  }
  count++;

  return write_ret;
}

__device__ void compbuf_t::release() {
#ifdef DEBUG
  assert(count > 0);
#endif

  if (++read == max) {
    read = 0;
  }
  count--;
}

__device__ void compbuf_t::print() {
  printf("[compbuf_t] ptr: %p, max: %llu, count: %llu, write: %lld, read: %lld\n",
      ptr, max, count, write, read);
}
