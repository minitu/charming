#ifndef _SCHEDULER_H_
#define _SCHEDULER_H_

#include "message.h"

namespace charm {

__device__ msgtype process_msg_pe(void* addr, size_t offset,
    bool& begin_term_flag, bool& do_term_flag);
__device__ msgtype process_msg_ce(void* addr, size_t offset,
    bool& begin_term_flag, bool& do_term_flag);
__global__ void scheduler(int argc, char** argv, size_t* argvs);

}

#endif // _SCHEDULER_H_
