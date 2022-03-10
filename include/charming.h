#ifndef _CHARMING_H_
#define _CHARMING_H_

#include "chare.h"

// Maximum number of messages that are allowed to be in flight per pair of PEs
#define REMOTE_MSG_COUNT_MAX 4
// Maximum number of messages that can be stored in the local message queue
#define LOCAL_MSG_COUNT_MAX 128

// Print functions
#define PINFO(...) printf("[INFO] " __VA_ARGS__)
#define PERROR(...) printf("[ERROR] " __VA_ARGS__)
#ifdef DEBUG
#define PDEBUG(...) printf("[DEBUG] " __VA_ARGS__)
#else
#define PDEBUG(...) do {} while (0)
#endif

namespace charm {

// User functions required by the runtime
__device__ void register_chares();
__device__ void main(int argc, char** argv, size_t* argvs);

// Runtime functions that can be called by the user
__device__ void end();

__device__ int my_pe();
__device__ int n_pes();
__device__ int my_pe_node();
__device__ int n_pes_node();
__device__ int n_nodes();

__device__ int device_atoi(const char* str, int strlen);

}

#endif
