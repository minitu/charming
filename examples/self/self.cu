#include <stdio.h>
#include "self.h"

__shared__ charm::chare_proxy<Comm>* comm_proxy;

__device__ void charm::create_chares(int argc, char** argv, size_t* argvs) {
  comm_proxy = new charm::chare_proxy<Comm>();
  comm_proxy->add_entry_method<&entry_init>();
  comm_proxy->add_entry_method<&entry_run>();
  comm_proxy->create(1);
}

__device__ void charm::main(int argc, char** argv, size_t* argvs) {
  constexpr int n_params = 4;
  __shared__ size_t params[n_params];

  if (threadIdx.x == 0) {
    // Default parameters
    params[0] = 1;
    params[1] = 65536;
    params[2] = 100;
    params[3] = 10;

    // Process command line arguments
    if (argc >= 2) params[0] = charm::device_atoi(argv[1], argvs[1]);
    if (argc >= 3) params[1] = charm::device_atoi(argv[2], argvs[2]);
    if (argc >= 4) params[2] = charm::device_atoi(argv[3], argvs[3]);
    if (argc >= 5) params[3] = charm::device_atoi(argv[4], argvs[4]);

    printf("Self messaging\n");
    printf("Size: %llu - %llu\n", params[0], params[1]);
    printf("Iterations: %d (Warmup: %d)\n", (int)params[2], (int)params[3]);
  }
  __syncthreads();

  comm_proxy->invoke_all(0, params, sizeof(size_t) * n_params);
}

__device__ void Comm::init(void* arg) {
  if (threadIdx.x == 0) {
    size_t* params = (size_t*)arg;
    int param_idx = 0;

    init_cnt = 0;
    min_size = params[param_idx++];
    max_size = params[param_idx++];
    cur_size = min_size;
    n_iters = static_cast<int>(params[param_idx++]);
    warmup = static_cast<int>(params[param_idx++]);
    iter = 0;
#ifdef USER_MSG
    msg.alloc(max_size);
#else
    data = new char[max_size];
#endif

    index = charm::chare::i;
    self = 0;
    printf("Chare %d init\n", index);
  }
  __syncthreads();

  comm_proxy->invoke(self, 1, data, cur_size);
}

__device__ void Comm::run(void* arg) {
  __shared__ bool end;

  if (threadIdx.x == 0) {
    end = false;

    if (iter == n_iters + warmup) {
      // End of iterations
      end_tp = cuda::std::chrono::system_clock::now();
      cuda::std::chrono::duration<double> diff = end_tp - start_tp;
      printf("Size %llu took %.3lf us\n", cur_size, (diff.count() / n_iters) * 1000000);
      cur_size *= 2;
      iter = 0;
      if (cur_size > max_size) {
        end = true;
      }
    }

    if (!end && iter < n_iters + warmup) {
      // Continue next iteration
      if (iter == warmup) {
        start_tp = cuda::std::chrono::system_clock::now();
      }
#ifdef DEBUG
      printf("Index %d iter %d size %lu\n", index, iter, cur_size);
#endif
    }
  }
  __syncthreads();

  if (end) {
    charm::end();
  } else {
#ifdef USER_MSG
    comm_proxy->invoke(self, 1, msg, cur_size);
#else
    comm_proxy->invoke(self, 1, data, cur_size);
#endif

    if (threadIdx.x == 0) {
      iter++;
    }
    __syncthreads();
  }
}
