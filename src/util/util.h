#ifndef _UTIL_H_
#define _UTIL_H_

#include <cuda/std/chrono>

#define cuda_check_error() {                                                         \
  cudaError_t e = cudaGetLastError();                                                \
  if (cudaSuccess != e) {                                                            \
    printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(-1);                                                                        \
  }                                                                                  \
}

using clock_value_t = long long;
using cudatp_t = cuda::std::chrono::time_point<cuda::std::chrono::system_clock>;
using cudatp_dur_t = cuda::std::chrono::duration<double>;

__device__ uint get_smid();
__device__ void sleep(clock_value_t sleep_cycles);

#endif // _UTIL_H_
