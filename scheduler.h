#ifndef _SCHEDULER_H_
#define _SCHEDULER_H_

#include "message.h"

namespace charm {

__device__ envelope* create_envelope(msgtype type, size_t msg_size);
__device__ void send_msg(envelope* env, size_t msg_size, int dst_pe);
__device__ void send_dummy_msg(int dst_pe);
__device__ void send_reg_msg(int chare_id, int chare_idx, int ep_id, size_t payload_size, int dst_pe);
__device__ void send_term_msg(int dst_pe);

__global__ void scheduler();
}

#endif // _SCHEDULER_H_
