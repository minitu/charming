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

__device__ msgtype charm::process_msg_pe(void* addr, size_t offset,
    bool& begin_term_flag, bool& do_term_flag) {
  int my_pe = s_mem[s_idx::my_pe];
  envelope* env = (envelope*)addr;
  msgtype type = env->type;
  if (threadIdx.x == 0) {
    PDEBUG("PE %d processing msg type %d size %llu\n", my_pe, type, env->size);
  }

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

  } else if (type == msgtype::forward) {
    // Forwarded from CE
    forward_msg* msg = (forward_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("PE %d forward msg chare ID %d chare idx %d EP ID %d\n", my_pe,
          msg->chare_id, msg->chare_idx, msg->ep_id);
    }

    chare_proxy_base*& chare_proxy = my_proxy_table.proxies[msg->chare_id];
    void* payload = (char*)msg + sizeof(forward_msg);

    chare_proxy->call(msg->chare_idx, msg->ep_id, payload);

  } else if (type == msgtype::do_terminate) {
    // Received message to terminate
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

__device__ msgtype charm::process_msg_ce(void* addr, size_t offset,
    bool& sent_term_flag, bool& begin_term_flag, bool& do_term_flag) {
  int my_ce = s_mem[s_idx::my_ce];
  envelope* env = (envelope*)addr;
  msgtype type = env->type;
  if (threadIdx.x == 0) {
    PDEBUG("CE %d processing msg type %d size %llu\n", my_ce, type, env->size);
  }

  // TODO: User message
  if (type == msgtype::request) {
    // Only CEs receive requests
    request_msg* msg = (request_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("CE %d request msg chare ID %d chare idx %d EP ID %d msgtype %d "
          "buf %p payload_size %llu dst PE %d\n", my_ce, msg->chare_id,
          msg->chare_idx, msg->ep_id, msg->type, msg->buf, msg->payload_size,
          msg->dst_pe);
    }

    if (msg->type == msgtype::begin_terminate) {
      // If a begin termination message has already been sent from this CE,
      // don't send it again
      if (sent_term_flag) return type;
      if (threadIdx.x == 0) {
        sent_term_flag = true;
      }
      __syncthreads();
    }

    // Send message to remote CE as delegate of source PE
    send_delegate_msg(msg);

  } else if (type == msgtype::forward) {
    // Message from another CE, need to forward to the right PE
    forward_msg* msg = (forward_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("CE %d forward msg offset %llu dst PE %d chare ID %d chare idx %d EP ID %d\n",
          my_ce, offset, msg->dst_pe, msg->chare_id, msg->chare_idx, msg->ep_id);
    }

    // Forward message to target PE
    int dst_local_rank = get_local_rank_from_pe(msg->dst_pe);
    send_local_msg(env, offset, dst_local_rank);

  } else if (type == msgtype::begin_terminate) {
    // Received begin termination message
    // Should only be received by CE 0
    assert(my_ce == 0);

    // Check if begin_terminate message was already received
    if (begin_term_flag) return type;

    // Send out do_terminate messages to all CEs
    if (threadIdx.x == 0) {
      PDEBUG("CE %d begin terminate msg\n", my_ce);
      begin_term_flag = true;
    }
    __syncthreads();

    for (int ce = 0; ce < c_n_ces; ce++) {
      send_do_term_msg_ce(ce);
    }

  } else if (type == msgtype::do_terminate) {
    // Send out do_terminate messages to all child PEs
    if (threadIdx.x == 0) {
      PDEBUG("CE %d do terminate msg\n", my_ce);
      do_term_flag = true;
    }
    __syncthreads();

    comm* c = (comm*)(s_mem + SMEM_CNT_MAX);
    for (int i = 0; i < c->child_count; i++) {
      send_do_term_msg_pe(c->child_local_ranks[i]);
    }

  } else {
    if (threadIdx.x == 0) {
      PERROR("CE %d unrecognized message type %d\n", my_ce, type);
    }
    assert(false);
  }

  return type;
}


__device__ __forceinline__ void loop_pe(comm* c) {
  c->process_local();
#ifndef NO_CLEANUP
  c->cleanup_local();
  c->cleanup_heap();
#endif
}

__device__ __forceinline__ void loop_ce(comm* c) {
  c->process_local();
  c->process_remote();
#ifndef NO_CLEANUP
  c->cleanup_local();
  c->cleanup_remote();
  c->cleanup_heap();
#endif
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
  if (s_mem[s_idx::is_pe]) {
    do {
      loop_pe(c);
    } while (!c->do_term_flag);
  } else {
    do {
      loop_ce(c);
    } while (!c->do_term_flag);
  }

  // Global synchronization
  if (gid == 0) {
    nvshmem_barrier_all();
  }
  grid.sync();

  if (threadIdx.x == 0) {
    PDEBUG("%s %d terminating...\n", s_mem[s_idx::is_pe] ? "PE" : "CE",
        s_mem[s_idx::is_pe] ? (int)s_mem[s_idx::my_pe] : (int)s_mem[s_idx::my_ce]);
  }
}
