#ifndef _COMM_H_
#define _COMM_H_

#include "heap.h"

namespace charm {

void comm_init_host(int n_pes);
void comm_fini_host();

struct comm {
  min_heap addr_heap;
  bool begin_term_flag;
  bool do_term_flag;

  __device__ comm();
  __device__ void process_local();
  __device__ void process_remote();
  __device__ void cleanup();
};

}

#endif // _COMM_H_
