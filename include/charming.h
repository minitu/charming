#ifndef _CHARMING_H_
#define _CHARMING_H_

#include "chare.h"
#include "message.h"
#include "kernel.h"

namespace charm {

// Main function executed on all host PEs
void main_host(int argc, char** argv);

// Main function executed on all PEs
__device__ void main(int argc, char** argv, size_t* argvs, int pe);

// Runtime functions that can be called by the user
__device__ void end();
__device__ void abort();

__device__ int my_pe();
__device__ int n_pes();
__device__ int my_pe_node();
__device__ int n_pes_node();
__device__ int n_nodes();

// Element-level barrier (requires all PEs and CEs to participate)
__device__ void barrier();
__device__ void barrier_local();

__device__ int device_atoi(const char* str, int strlen);

#ifndef SM_LEVEL
__device__ void* malloc_user(size_t size, size_t& offset);
__device__ void free_user(size_t size, size_t offset);
#endif

}

#endif // _CHARMING_H_
