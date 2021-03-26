#ifndef _NVCHARM_H_
#define _NVCHARM_H_

#include "chare.h"

namespace charm {

// User functions required by the runtime
__device__ void register_chare_types(chare_type** chare_types);
__device__ void main(chare_type** chare_types);

// Runtime functions that can be called by the user
__device__ void exit();

}

#endif
