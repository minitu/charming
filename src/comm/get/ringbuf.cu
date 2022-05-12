#include "ringbuf.h"
#include <nvshmem.h>
#include <cassert>

__host__ void ringbuf_t::init(void* ptr, size_t size, int my_sm) {
  base = ptr;
  start_offset = size * my_sm;
  space = size;
  read_end = UINT64_MAX;
  write = 0;
  read = 0;
}

__device__ bool ringbuf_t::acquire(size_t size, size_t& ret_offset) {
  // Compute potential new write offset
  size_t new_write = write + size;
  if (write < read && new_write >= read) {
    // Write should never catch up to read
    return false;
  }

  size_t write_ret = write;
  if (new_write >= space) {
    // Need to Wrap-around
#ifdef DEBUG
    assert(read <= write);
    assert(read_end == UINT64_MAX);
#endif
    read_end = write;
    const bool exceed = new_write > space;
    if (exceed) {
      // Early wrap-around
      new_write = size;
      write_ret = 0;
    } else {
      // Exact wrap-around
      new_write = 0;
    }

    // New write should never catch up to read
    if (new_write >= read) {
      return false;
    }
  }

  // If read is already at the end, move to beginning
  if (read == read_end) {
    read = 0;
  }

  // Update write value
#ifdef DEBUG
  assert(new_write <= space);
#endif
  write = new_write;

  // Prepare return offset
  ret_offset = start_offset + write_ret;

  return true;
}

__device__ void ringbuf_t::release(size_t size) {
#ifdef DEBUG
  assert((read <= write && (read + size <= write))
      || (read > write && (read + size <= read_end)));
#endif
  read += size;
  if (read == read_end) {
    read_end = UINT64_MAX;
    read = 0;
  }
}

__device__ void ringbuf_t::print() {
  printf("[ringbuf_t] base: %p, start_offset: %llu, space: %llu, read_end: %llu, "
      "write: %lld, read: %lld\n",
      base, start_offset, space, read_end, write, read);
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
