#ifndef _SCHEDULER_H_
#define _SCHEDULER_H_

#include "message.h"

namespace charm {

__device__ envelope* create_envelope(msgtype type, size_t msg_size, size_t* offset);
__device__ void send_msg(size_t offset, size_t msg_size, int dst_pe);
__device__ void send_reg_msg(int chare_id, int chare_idx, int ep_id, void* buf, size_t payload_size, int dst_pe);
__device__ void send_begin_term_msg(int dst_pe);
__device__ void send_do_term_msg(int dst_pe);

__global__ void scheduler(int argc, char** argv, size_t* argvs);

}

#endif // _SCHEDULER_H_
