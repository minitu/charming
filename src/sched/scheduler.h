#ifndef _SCHEDULER_H_
#define _SCHEDULER_H_

#include "message.h"

namespace charm {

__device__ msgtype process_msg(void* addr, ssize_t* processed_size,
    bool& begin_term_flag, bool& do_term_flag);
__global__ void scheduler(int argc, char** argv, size_t* argvs);

}

#endif // _SCHEDULER_H_
