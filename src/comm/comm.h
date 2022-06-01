#ifndef _COMM_H_
#define _COMM_H_

#define ALIGN_SIZE 16

#if CHARMING_COMM_TYPE == 0
#include "heap.h"
#endif

namespace charm {

#ifdef SM_LEVEL
void comm_init_host(int n_sms, int n_pes, int n_ces, int h_n_clusters_dev,
    int n_pes_cluster, int n_ces_cluster);
#else
void comm_init_host(int n_pes);
#endif
void comm_fini_host();

struct alignas(ALIGN_SIZE) comm {
#if CHARMING_COMM_TYPE == 0
  min_heap addr_heap;
#endif

  bool sent_term_flag;
  bool begin_term_flag;
  bool do_term_flag;

#ifdef SM_LEVEL
  int* child_local_ranks;
  int child_count;
#else
  // For communication between the leader TB and other TBs in the grid
  size_t count;
  void* env;
  void* dst_addr;
  size_t dst_offset;
  void* src_addr;
  size_t src_offset;

  // TODO: Support more than one chare array
  int async_wait_chare_id;
#endif

  int local_start;

  __device__ void init();
#ifdef SM_LEVEL
  __device__ void process_local();
  __device__ void cleanup_local();
#endif
  __device__ void process_remote();
  __device__ void cleanup_remote();
  __device__ void cleanup_heap();
#ifndef SM_LEVEL
  __device__ void check_async_wait();
#endif
};

#ifndef SM_LEVEL
__device__ void* malloc_user(size_t size, size_t& offset);
__device__ void free_user(size_t size, size_t offset);
#endif

} // namespace charm

#endif // _COMM_H_
