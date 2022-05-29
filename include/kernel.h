#ifndef _KERNEL_H_
#define _KERNEL_H_

namespace charm {

  __device__ __forceinline__ void memset_kernel_block(void* addr, int val, size_t size) {
    for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
      ((char*)addr)[i] = (char)val;
    }
  }

  __device__ __forceinline__ void memset_kernel_grid(void* addr, int val, size_t size) {
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
      ((char*)addr)[i] = (char)val;
    }
  }

  __device__ __forceinline__ void memcpy_kernel_block(void* dst, void* src, size_t size) {
    for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
      ((char*)dst)[i] = ((char*)src)[i];
    }
  }

  __device__ __forceinline__ void memcpy_kernel_grid(void* dst, void* src, size_t size) {
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
      ((char*)dst)[i] = ((char*)src)[i];
    }
  }

  __device__ inline void nvshmem_memcpy_block(void *__restrict__ dst,
      const void *__restrict__ src, size_t len) {
    int myIdx = threadIdx.x;
    int groupSize = blockDim.x;

    /*
     * If src and dst are 16B aligned copy as much as possible using 16B chunks
     */
    if ((uintptr_t) dst % 16 == 0 && (uintptr_t) src % 16 == 0) {
      int4 * __restrict__ dst_p =       (int4 *) dst;
      const int4 * __restrict__ src_p = (const int4 *) src;
      const size_t nelems = len / 16;

      for (size_t i = myIdx; i < nelems; i += groupSize)
        dst_p[i] = src_p[i];

      len -= nelems * 16;

      if (0 == len) return;

      dst = (void *) (dst_p + nelems);
      src = (void *) (src_p + nelems);
    }

    /*
     * If src and dst are 8B aligned copy as much as possible using 8B chunks
     */
    if ((uintptr_t) dst % 8 == 0 && (uintptr_t) src % 8 == 0) {
      uint64_t * __restrict__ dst_p =       (uint64_t *) dst;
      const uint64_t * __restrict__ src_p = (const uint64_t *) src;
      const size_t nelems = len / 8;

      for (size_t i = myIdx; i < nelems; i += groupSize)
        dst_p[i] = src_p[i];

      len -= nelems * 8;

      if (0 == len) return;

      dst = (void *) (dst_p + nelems);
      src = (void *) (src_p + nelems);
    }

    /*
     * If src and dst are 4B aligned copy as much as possible using 4B chunks
     */
    if ((uintptr_t) dst % 4 == 0 && (uintptr_t) src % 4 == 0) {
      uint32_t * __restrict__ dst_p =       (uint32_t *) dst;
      const uint32_t * __restrict__ src_p = (const uint32_t *) src;
      const size_t nelems = len / 4;

      for (size_t i = myIdx; i < nelems; i += groupSize)
        dst_p[i] = src_p[i];

      len -= nelems * 4;

      if (0 == len) return;

      dst = (void *) (dst_p + nelems);
      src = (void *) (src_p + nelems);
    }

    /*
     * If src and dst are 2B aligned copy as much as possible using 2B chunks
     */
    if ((uintptr_t) dst % 2 == 0 && (uintptr_t) src % 2 == 0) {
      uint16_t * __restrict__ dst_p =       (uint16_t *) dst;
      const uint16_t * __restrict__ src_p = (const uint16_t *) src;
      const size_t nelems = len / 2;

      for (size_t i = myIdx; i < nelems; i += groupSize)
        dst_p[i] = src_p[i];

      len -= nelems * 2;

      if (0 == len) return;

      dst = (void *) (dst_p + nelems);
      src = (void *) (src_p + nelems);
    }

    unsigned char * __restrict__ dst_c =       (unsigned char *) dst;
    const unsigned char * __restrict__ src_c = (const unsigned char *) src;

    for (size_t i = myIdx; i < len; i += groupSize)
      dst_c[i] = src_c[i];
  }

  __device__ inline void nvshmem_memcpy_grid(void *__restrict__ dst,
      const void *__restrict__ src, size_t len) {
    int myIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int groupSize = gridDim.x * blockDim.x;

    /*
     * If src and dst are 16B aligned copy as much as possible using 16B chunks
     */
    if ((uintptr_t) dst % 16 == 0 && (uintptr_t) src % 16 == 0) {
      int4 * __restrict__ dst_p =       (int4 *) dst;
      const int4 * __restrict__ src_p = (const int4 *) src;
      const size_t nelems = len / 16;

      for (size_t i = myIdx; i < nelems; i += groupSize)
        dst_p[i] = src_p[i];

      len -= nelems * 16;

      if (0 == len) return;

      dst = (void *) (dst_p + nelems);
      src = (void *) (src_p + nelems);
    }

    /*
     * If src and dst are 8B aligned copy as much as possible using 8B chunks
     */
    if ((uintptr_t) dst % 8 == 0 && (uintptr_t) src % 8 == 0) {
      uint64_t * __restrict__ dst_p =       (uint64_t *) dst;
      const uint64_t * __restrict__ src_p = (const uint64_t *) src;
      const size_t nelems = len / 8;

      for (size_t i = myIdx; i < nelems; i += groupSize)
        dst_p[i] = src_p[i];

      len -= nelems * 8;

      if (0 == len) return;

      dst = (void *) (dst_p + nelems);
      src = (void *) (src_p + nelems);
    }

    /*
     * If src and dst are 4B aligned copy as much as possible using 4B chunks
     */
    if ((uintptr_t) dst % 4 == 0 && (uintptr_t) src % 4 == 0) {
      uint32_t * __restrict__ dst_p =       (uint32_t *) dst;
      const uint32_t * __restrict__ src_p = (const uint32_t *) src;
      const size_t nelems = len / 4;

      for (size_t i = myIdx; i < nelems; i += groupSize)
        dst_p[i] = src_p[i];

      len -= nelems * 4;

      if (0 == len) return;

      dst = (void *) (dst_p + nelems);
      src = (void *) (src_p + nelems);
    }

    /*
     * If src and dst are 2B aligned copy as much as possible using 2B chunks
     */
    if ((uintptr_t) dst % 2 == 0 && (uintptr_t) src % 2 == 0) {
      uint16_t * __restrict__ dst_p =       (uint16_t *) dst;
      const uint16_t * __restrict__ src_p = (const uint16_t *) src;
      const size_t nelems = len / 2;

      for (size_t i = myIdx; i < nelems; i += groupSize)
        dst_p[i] = src_p[i];

      len -= nelems * 2;

      if (0 == len) return;

      dst = (void *) (dst_p + nelems);
      src = (void *) (src_p + nelems);
    }

    unsigned char * __restrict__ dst_c =       (unsigned char *) dst;
    const unsigned char * __restrict__ src_c = (const unsigned char *) src;

    for (size_t i = myIdx; i < len; i += groupSize)
      dst_c[i] = src_c[i];
  }


}; // namespace charm

#endif // _KERNEL_H_
