#ifndef _SCHEDULER_H_
#define _SCHEDULER_H_

namespace charm {

__device__ ssize_t process_msg(void* addr, bool& begin_term_flag, bool& do_term_flag);
__global__ void scheduler(int argc, char** argv, size_t* argvs);

}

#endif // _SCHEDULER_H_
