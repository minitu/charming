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

__device__ ssize_t charm::process_msg(void* addr, bool& begin_term_flag, bool& do_term_flag) {
  envelope* env = (envelope*)addr;
  PDEBUG("PE %d: received msg type %d size %llu from PE %d\n",
         c_my_pe, env->type, env->size, env->src_pe);

  if (env->type == msgtype::create) {
    // Creation message
    create_msg* msg = (create_msg*)((char*)env + sizeof(envelope));
    PDEBUG("PE %d: creation msg chare ID %d, n_local %d, n_total %d, "
        "start idx %d, end idx %d\n", c_my_pe, msg->chare_id, msg->n_local,
        msg->n_total, msg->start_idx, msg->end_idx);

    chare_proxy_base*& chare_proxy = chare_proxies[msg->chare_id];
    char* map_ptr = (char*)msg + sizeof(create_msg);
    char* obj_ptr = map_ptr + sizeof(int) * msg->n_total;

    chare_proxy->create_local(msg->n_local, msg->n_total, msg->start_idx,
        msg->end_idx, map_ptr, obj_ptr);

  } else if (env->type == msgtype::regular) {
    // Regular message
    regular_msg* msg = (regular_msg*)((char*)env + sizeof(envelope));
    PDEBUG("PE %d: regular msg chare ID %d chare idx %d EP ID %d\n", c_my_pe,
        msg->chare_id, msg->chare_idx, msg->ep_id);

    chare_proxy_base*& chare_proxy = chare_proxies[msg->chare_id];
    void* payload = (char*)msg + sizeof(regular_msg);

    chare_proxy->call(msg->chare_idx, msg->ep_id, payload);

  } else if (env->type == msgtype::begin_terminate) {
    // Should only be received by PE 0
    assert(my_pe() == 0);

    // Begin termination message
    PDEBUG("PE %d: begin terminate msg\n", c_my_pe);
    if (!begin_term_flag) {
      for (int i = 0; i < n_pes(); i++) {
        send_do_term_msg(i);
      }
      begin_term_flag = true;
    }

  } else if (env->type == msgtype::do_terminate) {
    // Do termination message
    PDEBUG("PE %d: do terminate msg\n", c_my_pe);
    do_term_flag = true;

  } else {
    PERROR("PE %d: unrecognized message type %d\n", c_my_pe, env->type);
    assert(false);
  }

  return env->size;
}

__device__ __forceinline__ void loop(comm& c) {
  c.process_local();
  c.process_remote();
  c.cleanup();
}

__global__ void charm::scheduler(int argc, char** argv, size_t* argvs) {
  if (!blockIdx.x && !threadIdx.x) {
    // Create communication module on the stack
    comm c;

    // Register user chares and entry methods on all PEs
    chare_proxy_cnt = 0;
    register_chares();

    nvshmem_barrier_all();

    if (c_my_pe == 0) {
      // Execute user's main function
      main(argc, argv, argvs);
    }

    // FIXME: Is this barrier necessary?
    nvshmem_barrier_all();

    // Loop until termination
    do {
      loop(c);
    } while (!c.do_term_flag);

    PDEBUG("PE %d terminating...\n", c_my_pe);
  }
}
