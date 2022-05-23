#include <stdio.h>
#include "refnum.h"

__shared__ charm::chare_proxy<Comm>* comm_proxy;

__device__ void charm::main(int argc, char** argv, size_t* argvs, int pe) {
  // Execute on all elements
  if (threadIdx.x == 0) {
    // Create chare proxy and register entry methods
    comm_proxy = new charm::chare_proxy<Comm>();
    comm_proxy->add_entry_method<&entry_send>();
    comm_proxy->add_entry_method<&entry_recv>();

    // Custom block map to pick 2 PEs on the opposite sides
    int n_pes = charm::n_pes();
    int* block_map = new int[n_pes];
    for (int i = 0; i < n_pes; i++) {
      block_map[i] = 0;
    }
    block_map[0] = 1;
    block_map[n_pes/2] = 1;

    // Create chares
    comm_proxy->create(2, block_map);

    // Set initial reference number per chare
    if (pe == 0) comm_proxy->set_refnum(0, 0);
    if (pe == n_pes/2) comm_proxy->set_refnum(1, 1);
  }
  __syncthreads();

  barrier();

  // Execute only on PE 0
  if (pe == 0) {
    comm_proxy->invoke(0, 0, nullptr, 0, 0);
  }
}

__device__ void Comm::send(void* arg) {
  int refnum = 1;
  comm_proxy->invoke(1, 1, &refnum, sizeof(int), refnum);
  refnum = 0;
  comm_proxy->invoke(1, 1, &refnum, sizeof(int), refnum);
}

__device__ void Comm::recv(void* arg) {
  if (threadIdx.x == 0) {
    printf("Received message with refnum %d\n", *(int*)arg);
    comm_proxy->set_refnum(1, 0);
  }
  __syncthreads();
}
