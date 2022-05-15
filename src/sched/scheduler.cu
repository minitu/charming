#include <nvshmem.h>
#include <nvshmemx.h>
#include <cooperative_groups.h>

#include "charming.h"
#include "common.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "chare.h"
#include "util.h"

namespace cg = cooperative_groups;
using namespace charm;

// GPU global memory
extern __device__ __managed__ chare_proxy_table* proxy_tables;

// GPU shared memory
extern __shared__ uint64_t s_mem[];

__device__ msgtype charm::process_msg_pe(void* addr, ssize_t* processed_size,
    bool& begin_term_flag, bool& do_term_flag) {
  int my_pe = s_mem[s_idx::my_pe];
  envelope* env = (envelope*)addr;
  msgtype type = env->type;
  if (threadIdx.x == 0) {
    if (processed_size) *processed_size = env->size;
    PDEBUG("PE %d processing msg type %d size %llu\n", my_pe, type, env->size);
  }
  __syncthreads();

  chare_proxy_table& my_proxy_table = proxy_tables[blockIdx.x];

  if (type == msgtype::regular || type == msgtype::user) {
    // Regular message (including user message)
    regular_msg* msg = (regular_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("PE %d regular msg chare ID %d chare idx %d EP ID %d\n", my_pe,
          msg->chare_id, msg->chare_idx, msg->ep_id);
    }

    chare_proxy_base*& chare_proxy = my_proxy_table.proxies[msg->chare_id];
    void* payload = (char*)msg + sizeof(regular_msg);

    chare_proxy->call(msg->chare_idx, msg->ep_id, payload);

  } else if (type == msgtype::begin_terminate) {
    // TODO
    // Should only be received by PE 0
    assert(my_pe == 0);

    // Begin termination message
    if (!begin_term_flag) {
      if (threadIdx.x == 0) {
        PDEBUG("PE %d begin terminate msg\n", my_pe);
        begin_term_flag = true;
      }
      __syncthreads();

      for (int pe = 0; pe < c_n_pes; pe++) {
        send_term_msg(false, pe);
      }
    }

  } else if (type == msgtype::do_terminate) {
    // TODO
    // Do termination message
    if (threadIdx.x == 0) {
      PDEBUG("PE %d do terminate msg\n", my_pe);
      do_term_flag = true;
    }
    __syncthreads();

  } else {
    if (threadIdx.x == 0) {
      PERROR("PE %d unrecognized message type %d\n", my_pe, type);
    }
    assert(false);
  }

  return type;
}

__device__ msgtype charm::process_msg_ce(void* addr, ssize_t* processed_size,
    bool& begin_term_flag, bool& do_term_flag) {
  int my_ce = s_mem[s_idx::my_ce];
  envelope* env = (envelope*)addr;
  msgtype type = env->type;
  if (threadIdx.x == 0) {
    if (processed_size) *processed_size = env->size;
    PDEBUG("CE %d processing msg type %d size %llu", my_ce, type, env->size);
  }
  __syncthreads();

  chare_proxy_table& my_proxy_table = proxy_tables[blockIdx.x];

  if (type == msgtype::regular || type == msgtype::user) {
    // TODO
    // Regular message (including user message)
    regular_msg* msg = (regular_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("PE %d regular msg chare ID %d chare idx %d EP ID %d\n", my_pe,
          msg->chare_id, msg->chare_idx, msg->ep_id);
    }

    chare_proxy_base*& chare_proxy = my_proxy_table.proxies[msg->chare_id];
    void* payload = (char*)msg + sizeof(regular_msg);

    chare_proxy->call(msg->chare_idx, msg->ep_id, payload);

  } else if (type == msgtype::request) {
    // Only CEs would receive requests
    request_msg* msg = (request_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("CE %d request msg chare ID %d chare idx %d EP ID %d msgtype %d"
          "buf %p payload_size %llu dst PE %d\n", my_ce, msg->chare_id,
          msg->chare_idx, msg->ep_id, msg->type, msg->buf, msg->payload_size,
          msg->dst_pe);
    }

    // Send remote message to CE responsible for target PE
    send_ce_msg(msg);

  } else if (type == msgtype::begin_terminate) {
    // TODO
    // Should only be received by PE 0
    assert(my_pe == 0);

    // Begin termination message
    if (!begin_term_flag) {
      if (threadIdx.x == 0) {
        PDEBUG("PE %d begin terminate msg\n", my_pe);
        begin_term_flag = true;
      }
      __syncthreads();

      for (int pe = 0; pe < c_n_pes; pe++) {
        send_term_msg(false, pe);
      }
    }

  } else if (type == msgtype::do_terminate) {
    // TODO
    // Do termination message
    if (threadIdx.x == 0) {
      PDEBUG("PE %d do terminate msg\n", my_pe);
      do_term_flag = true;
    }
    __syncthreads();

  } else {
    if (threadIdx.x == 0) {
      PERROR("PE %d unrecognized message type %d\n", my_pe, type);
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
  // For grid synchronization
  cg::grid_group grid = cg::this_grid();

  // Communication module resides in shared memory (one per TB)
  comm* c = (comm*)(s_mem + SMEM_CNT_MAX);

  if (threadIdx.x == 0) {
    // Store my PE or CE number in shared memory
    int my_cluster = blockIdx.x / c_cluster_size;
    int my_rank_in_cluster = blockIdx.x % c_cluster_size;
    int is_pe = (my_rank_in_cluster < c_n_pes_cluster) ? 1 : 0;
    int my_pe = (c_my_dev * c_n_clusters_dev + my_cluster) * c_n_pes_cluster
      + my_rank_in_cluster;
    int my_ce = (c_my_dev * c_n_clusters_dev + my_cluster) * c_n_ces_cluster
      + my_rank_in_cluster - c_n_pes_cluster;
    s_mem[s_idx::is_pe] = (uint64_t)is_pe;
    if (is_pe) {
      s_mem[s_idx::my_pe] = my_pe;
      s_mem[s_idx::my_ce] = UINT64_MAX;
    } else {
      s_mem[s_idx::my_pe] = UINT64_MAX;
      s_mem[s_idx::my_ce] = my_ce;
    }

    // Initialize comm module
    c->init();

    // Create user chares and register entry methods
    create_chares(argc, argv, argvs);
  }
  __syncthreads();

  // Global synchronization
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid == 0) {
    nvshmem_barrier_all();
  }
  grid.sync();

  /*
  int my_pe = s_mem[s_idx::my_pe];
  if (my_pe == 0) {
    // Execute user's main function
    main(argc, argv, argvs);
  }

  // Global synchronization
  if (gid == 0) {
    nvshmem_barrier_all();
  }
  grid.sync();

  // Loop until termination
  do {
    loop(c);
  } while (!c->do_term_flag);
  */

  // Global synchronization
  if (gid == 0) {
    nvshmem_barrier_all();
  }
  grid.sync();

  if (threadIdx.x == 0) {
    PDEBUG("PE %d terminating...\n", my_pe);
  }
}
