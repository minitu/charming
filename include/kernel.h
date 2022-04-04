#ifndef _KERNEL_H_
#define _KERNEL_H_

namespace charm {

  __device__ __forceinline__ void memset_kernel(void* addr, int val, size_t size) {
    for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
      ((char*)addr)[i] = (char)val;
    }
  }

  __device__ __forceinline__ void memcpy_kernel(void* dst, void* src, size_t size) {
    for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
      ((char*)dst)[i] = ((char*)src)[i];
    }
  }

}; // namespace charm

#endif // _KERNEL_H_
