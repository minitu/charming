#ifndef _SCHEDULER_H_
#define _SCHEDULER_H_

#include "message.h"

__device__ Envelope* create_envelope(MsgType type, size_t msg_size);
__device__ void send_msg(Envelope* env, size_t msg_size, int dst_pe);
__device__ void send_term_msg(int dst_pe);
__device__ void send_reg_msg(int chare_id, int ep_id, size_t payload_size, int dst_pe);

__global__ void scheduler();

#endif // _SCHEDULER_H_
