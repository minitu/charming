#ifndef _SCHEDULER_H_
#define _SCHEDULER_H_

#include "message.h"

namespace charm {

#ifdef SM_LEVEL
__device__ bool process_msg_pe(void* addr, size_t offset, msgtype& type,
    bool& begin_term_flag, bool& do_term_flag);
__device__ bool process_msg_ce(void* addr, size_t offset, msgtype& type,
    bool& sent_term_flag, bool& begin_term_flag, bool& do_term_flag);
#else
__device__ bool process_msg(void* addr, size_t offset, msgtype& type,
    bool& begin_term_flag, bool& do_term_flag);
#endif
__global__ void scheduler(int argc, char** argv, size_t* argvs);
__device__ void scheduler_barrier();
__device__ void scheduler_barrier_local();

}

#endif // _SCHEDULER_H_
