#include <nvshmem.h>
#include <nvshmemx.h>
#include <cooperative_groups.h>
#ifdef MEASURE_TIME
#include <cuda/std/chrono>
#endif

#include "charming.h"
#include "common.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "chare.h"
#include "util.h"

namespace cg = cooperative_groups;
using namespace charm;

// GPU constant memory
extern __constant__ int c_n_sms;
extern __constant__ int c_my_dev;
extern __constant__ int c_my_dev_node;
extern __constant__ int c_n_devs;
extern __constant__ int c_n_devs_node;
extern __constant__ int c_n_nodes;
extern __constant__ int c_n_pes;
extern __constant__ int c_n_pes_node;

// GPU global memory
extern __device__ __managed__ chare_proxy_table* proxy_tables;

// GPU shared memory
extern __shared__ uint64_t s_mem[];

__device__ msgtype charm::process_msg(void* addr, ssize_t* processed_size,
    bool& begin_term_flag, bool& do_term_flag) {
  int my_pe = s_mem[s_idx::my_pe];
  envelope* env = (envelope*)addr;
  msgtype type = env->type;
  if (threadIdx.x == 0) {
    if (processed_size) *processed_size = env->size;
    PDEBUG("PE %d received msg type %d size %llu from PE %d\n",
           my_pe, type, env->size, env->src_pe);
  }
  __syncthreads();

  chare_proxy_table& my_proxy_table = proxy_tables[blockIdx.x];
  if (type == msgtype::create) {
    // Creation message
    create_msg* msg = (create_msg*)((char*)env + sizeof(envelope));
    if (threadIdx.x == 0) {
      PDEBUG("PE %d creation msg chare ID %d, n_local %d, n_total %d, "
          "start idx %d, end idx %d\n", my_pe, msg->chare_id, msg->n_local,
          msg->n_total, msg->start_idx, msg->end_idx);
    }

    chare_proxy_base*& chare_proxy = my_proxy_table.proxies[msg->chare_id];
    char* map_ptr = (char*)msg + sizeof(create_msg);
    char* obj_ptr = map_ptr + sizeof(int) * msg->n_total;

    chare_proxy->create_local(msg->n_local, msg->n_total, msg->start_idx,
        msg->end_idx, map_ptr, obj_ptr);

  } else if (type == msgtype::regular || type == msgtype::user) {
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
#ifdef MEASURE_TIME
  c->n_iters++;
  auto start = cuda::std::chrono::system_clock::now();
#endif
  c->process_local();
#ifdef MEASURE_TIME
  auto end = cuda::std::chrono::system_clock::now();
  cuda::std::chrono::duration<double> diff = end - start;
  c->local_time += diff.count();
  start = cuda::std::chrono::system_clock::now();
#endif
  c->process_remote();
#ifdef MEASURE_TIME
  end = cuda::std::chrono::system_clock::now();
  diff = end - start;
  c->remote_time += diff.count();
  start = cuda::std::chrono::system_clock::now();
#endif
  c->cleanup();
#ifdef MEASURE_TIME
  end = cuda::std::chrono::system_clock::now();
  diff = end - start;
  c->cleanup_time += diff.count();
#endif
}

__global__ void charm::scheduler(int argc, char** argv, size_t* argvs) {
  // For grid synchronization
  cg::grid_group grid = cg::this_grid();

  // Communication module resides in shared memory (one per PE/TB)
  comm* c = (comm*)(s_mem + SMEM_CNT_MAX);

  if (threadIdx.x == 0) {
    // Register user chares and entry methods
    register_chares();

    // Initialize comm module
    c->init();

    // Store my PE number in shared memory
    s_mem[s_idx::my_pe] = c_my_dev * c_n_sms + blockIdx.x; // CharminG PE
    s_mem[s_idx::my_pe_node] = c_my_dev_node * c_n_sms + blockIdx.x; // CharminG PE rank on physical node
    s_mem[s_idx::my_pe_nvshmem] = s_mem[s_idx::my_pe] / c_n_sms; // NVSHMEM PE
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
  do {
    loop(c);
  } while (!c->do_term_flag);

  // Global synchronization
  if (gid == 0) {
    nvshmem_barrier_all();
  }
  grid.sync();

  if (threadIdx.x == 0) {
#ifdef MEASURE_TIME
    PINFO("PE %d scheduler time breakdown - local: %.3lf us, remote: %.3lf us, cleanup: %.3lf us\n",
        my_pe, c->local_time / c->n_iters * 1e6, c->remote_time / c->n_iters * 1e6,
        c->cleanup_time / c->n_iters * 1e6);
#endif
    PDEBUG("PE %d terminating...\n", my_pe);
  }
}
