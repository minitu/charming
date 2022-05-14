#ifndef _COMM_H_
#define _COMM_H_

#define ALIGN_SIZE 16

#if CHARMING_COMM_TYPE == 0
#include "heap.h"
#endif

namespace charm {

void comm_init_host(int n_pes, int n_sms, int h_n_clusters);
void comm_fini_host();

struct alignas(ALIGN_SIZE) comm {
#if CHARMING_COMM_TYPE == 0
  min_heap addr_heap_local;
  min_heap addr_heap_remote;
#endif

  bool begin_term_flag;
  bool do_term_flag;

  __device__ void init();
  __device__ void process_local();
  __device__ void process_remote();
  __device__ void cleanup();
};

}

#endif // _COMM_H_
