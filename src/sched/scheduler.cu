#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "chare.h"
#include "util.h"

using namespace charm;

extern __constant__ int c_my_pe;
extern __constant__ int c_n_pes;

extern __device__ chare_proxy_base* chare_proxies[];

extern __shared__ uint64_t s_mem[];

__device__ msgtype charm::process_msg(void* addr, ssize_t* processed_size,
    bool& begin_term_flag, bool& do_term_flag) {
  envelope* env = (envelope*)addr;
  msgtype type = env->type;
  if (threadIdx.x == 0) {
    if (processed_size) *processed_size = env->size;
    PDEBUG("PE %d: received msg type %d size %llu from PE %d\n",
           c_my_pe, type, env->size, env->src_pe);
  }
  __syncthreads();

  if (type == msgtype::create) {
    // Creation message
    create_msg* msg = (create_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("PE %d: creation msg chare ID %d, n_local %d, n_total %d, "
          "start idx %d, end idx %d\n", c_my_pe, msg->chare_id, msg->n_local,
          msg->n_total, msg->start_idx, msg->end_idx);
    }

    chare_proxy_base*& chare_proxy = chare_proxies[msg->chare_id];
    char* map_ptr = (char*)msg + sizeof(create_msg);
    char* obj_ptr = map_ptr + sizeof(int) * msg->n_total;

    chare_proxy->create_local(msg->n_local, msg->n_total, msg->start_idx,
        msg->end_idx, map_ptr, obj_ptr);

  } else if (type == msgtype::regular || type == msgtype::user) {
    // Regular message (including user message)
    regular_msg* msg = (regular_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("PE %d: regular msg chare ID %d chare idx %d EP ID %d\n", c_my_pe,
          msg->chare_id, msg->chare_idx, msg->ep_id);
    }

    chare_proxy_base*& chare_proxy = chare_proxies[msg->chare_id];
    void* payload = (char*)msg + sizeof(regular_msg);

    chare_proxy->call(msg->chare_idx, msg->ep_id, payload);

  } else if (type == msgtype::begin_terminate) {
    // Should only be received by PE 0
    assert(c_my_pe == 0);

    // Begin termination message
    if (threadIdx.x == 0) {
    PDEBUG("PE %d: begin terminate msg\n", c_my_pe);
      if (!begin_term_flag) {
        for (int i = 0; i < n_pes(); i++) {
          send_do_term_msg(i);
        }
        begin_term_flag = true;
      }
    }
    __syncthreads();

  } else if (type == msgtype::do_terminate) {
    // Do termination message
    if (threadIdx.x == 0) {
      PDEBUG("PE %d: do terminate msg\n", c_my_pe);
      do_term_flag = true;
    }
    __syncthreads();

  } else {
    if (threadIdx.x == 0) {
      PERROR("PE %d: unrecognized message type %d\n", c_my_pe, type);
    }
    assert(false);
  }

  return type;
}

__device__ __forceinline__ void loop(comm* c) {
  c->process_local();
  c->process_remote();
  c->cleanup();
}

__global__ void charm::scheduler(int argc, char** argv, size_t* argvs) {
  // TODO: Currently only 1 thread block per GPU
  if (blockIdx.x == 0) {
    // Communication module resides in shared memory
    // 32 bytes from start of shared memory for alignment
    comm* c = (comm*)(s_mem+4);

    if (threadIdx.x == 0) {
      // Initialize comm module
      c->init();

      // Register user chares and entry methods on all PEs
      chare_proxy_cnt = 0;
      register_chares();
      nvshmem_barrier_all();
    }
    __syncthreads();

    if (c_my_pe == 0) {
      // Execute user's main function
      main(argc, argv, argvs);
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      nvshmem_barrier_all();
    }
    __syncthreads();

    // Loop until termination
    do {
      loop(c);
    } while (!c->do_term_flag);

    if (threadIdx.x == 0) {
      PDEBUG("PE %d terminating...\n", c_my_pe);
    }
  }
}
