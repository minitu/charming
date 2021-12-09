#ifndef _COMPOSITE_H_
#define _COMPOSITE_H_

#include <stdio.h>
#include <cstdint>

#define OFFSET_BITS 36
#define SIZE_BITS   28
#define OFFSET_MASK 0xFFFFFFFFF0000000 // 36 bits
#define SIZE_MASK   0x000000000FFFFFFF // 28 bits

// Stores both the offset and size of a buffer
// by splitting a 64-bit unsigned integer. Used for
// Sending both values as a single NVSHMEM signal operation.
struct composite_t {
  uint64_t data;

  __device__ composite_t(uint64_t data_) {
    data = data_;
  }

  __device__ composite_t(size_t offset, size_t size) {
     data = static_cast<uint64_t>(offset);
     data <<= SIZE_BITS;
     data |= static_cast<uint64_t>(size);
  }

  __device__ size_t offset() const {
    return (data & OFFSET_MASK) >> SIZE_BITS;
  }

  __device__ size_t size() const {
    return data & SIZE_MASK;
  }

  friend __device__ bool operator<(const composite_t& lhs, const composite_t& rhs) {
    return lhs.offset() < rhs.offset();
  }
};

#endif // _COMPOSITE_H_
