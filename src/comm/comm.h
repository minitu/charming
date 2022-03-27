#ifndef _COMM_H_
#define _COMM_H_

#define ALIGN_SIZE 16

#if CHARMING_COMM_TYPE == 0
#include "heap.h"
#endif

namespace charm {

void comm_init_host(int n_pes);
void comm_fini_host(int n_pes);

struct alignas(ALIGN_SIZE) comm {
#if CHARMING_COMM_TYPE == 0
  min_heap addr_heap;
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
