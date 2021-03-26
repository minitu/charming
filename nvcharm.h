#ifndef _NVCHARM_H_
#define _NVCHARM_H_

#include "chare.h"

#define cuda_check_error() {                                                         \
  cudaError_t e = cudaGetLastError();                                                \
  if (cudaSuccess != e) {                                                            \
    printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(-1);                                                                        \
  }                                                                                  \
}

// User functions required by the runtime
__device__ void register_chare_types(ChareType** chare_types);
__device__ void charm_main(ChareType** chare_types);

// Runtime functions that can be called by the user
__device__ void ckExit();

#endif
