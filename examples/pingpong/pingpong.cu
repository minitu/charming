#include <stdio.h>
#include "pingpong.h"

__device__ charm::chare_proxy<Comm>* comm_proxy;

__device__ void charm::register_chares() {
  comm_proxy = new charm::chare_proxy<Comm>(3);
  comm_proxy->add_entry_method(&Comm::init);
  comm_proxy->add_entry_method(&Comm::init_done);
  comm_proxy->add_entry_method(&Comm::recv);
}

__device__ void Comm::init(void* arg) {
  size_t* params = (size_t*)arg;
  int param_idx = 0;

  init_cnt = 0;
  min_size = params[param_idx++];
  max_size = params[param_idx++];
  cur_size = min_size;
  n_iters = params[param_idx++];
  iter = 0;
  data = new char[max_size];

  index = charm::chare::i;
  peer = (index == 0) ? 1 : 0;
  printf("Chare %d init\n", index);

  comm_proxy->invoke(0, 1);
}

__device__ void Comm::init_done(void* arg) {
  if (++init_cnt == 2) {
    printf("Init done\n");

    start_tp = cuda::std::chrono::system_clock::now();
    send();
  }
}

__device__ void Comm::send() {
  comm_proxy->invoke(peer, 2, data, cur_size);
}

__device__ void Comm::recv(void* arg) {
  if (index == 0) {
    if (++iter == n_iters) {
      end_tp = cuda::std::chrono::system_clock::now();
      cuda::std::chrono::duration<double> diff = end_tp - start_tp;
      printf("Size %llu took %.3lf us\n", cur_size, (diff.count() / 2 / n_iters) * 1000000);
      start_tp = cuda::std::chrono::system_clock::now();
      cur_size *= 2;
      iter = 0;
      if (cur_size > max_size) {
        charm::end();
        return;
      }
    }
  }

  send();
}

// Main
__device__ void charm::main(int argc, char** argv, size_t* argvs) {
  if (charm::n_pes() != 2) {
    printf("Need exactly 2 PEs!\n");
    charm::end();
    return;
  }

  // Process command line arguments
  size_t min_size = 1;
  if (argc >= 2) min_size = charm::device_atoi(argv[1], argvs[1]);
  size_t max_size = 4194304;
  if (argc >= 3) max_size = charm::device_atoi(argv[2], argvs[2]);
  int n_iters = 100;
  if (argc >= 4) n_iters = charm::device_atoi(argv[3], argvs[3]);

  printf("Pingpong\n");
  printf("Size: %llu - %llu\n", min_size, max_size);
  printf("Iterations: %d\n", n_iters);

  Comm comm;
  comm_proxy->create(comm, 2);
  constexpr int n_params = 3;
  size_t params[n_params] = { min_size, max_size, n_iters };
  comm_proxy->invoke_all(0, params, sizeof(size_t) * n_params);
}
