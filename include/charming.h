#ifndef _CHARMING_H_
#define _CHARMING_H_

#include "chare.h"

#define DUMMY_ITERS 100000

namespace charm {

// User functions required by the runtime
__device__ void register_chares();
__device__ void main(int argc, char** argv, size_t* argvs);

// Runtime functions that can be called by the user
__device__ void exit();
__device__ int n_pes();
__device__ int my_pe();

}

#endif
