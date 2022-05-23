#include <stdio.h>
#include "hello.h"

__shared__ charm::chare_proxy<Hello>* hello_proxy;

__device__ void charm::main(int argc, char** argv, size_t* argvs, int pe) {
  // Execute on all elements
  if (threadIdx.x == 0) {
    // Register Hello chare and its entry methods
    hello_proxy = new charm::chare_proxy<Hello>();
    hello_proxy->add_entry_method<&entry_greet>();
    hello_proxy->create(charm::n_pes() * 2);
  }
  __syncthreads();

  barrier();

  // Execute only on PE 0
  if (pe == 0) {
    int* send_int_p = nullptr;
    if (threadIdx.x == 0) {
      send_int_p = new int;
      *send_int_p = 0;
    }
    __syncthreads();

    // Send integer to first chare
    hello_proxy->invoke(0, 0, send_int_p, sizeof(int));
  }
}

__device__ void Hello::greet(void* arg) {
  int recv_int = *(int*)arg;
  int* send_int_p;
  int i = charm::chare::i;
  int n = charm::chare::n;
  if (threadIdx.x == 0) {
    printf("Hello I'm %d of %d! Received %d\n", i, n, recv_int);
  }

  if (i == n-1) {
    charm::end();
  } else {
    if (threadIdx.x == 0) {
      send_int_p = new int;
      *send_int_p = recv_int + 1;
    }
    __syncthreads();

    hello_proxy->invoke(i + 1, 0, send_int_p, sizeof(int));
  }
}
