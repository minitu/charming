#include <stdio.h>
#include "pingpong.h"

#ifdef SM_LEVEL
// SM-level scheduling
#define GID (threadIdx.x)
#define BARRIER_LOCAL __syncthreads()
__shared__ charm::chare_proxy<Comm>* comm_proxy;
#else
// GPU-level scheduling
#define GID (blockDim.x * blockIdx.x + threadIdx.x)
#define BARRIER_LOCAL charm::barrier_local()
__device__ charm::chare_proxy<Comm>* comm_proxy;
__device__ size_t* params;
#endif

__device__ void charm::main(int argc, char** argv, size_t* argvs, int pe) {
  // Execute on all elements
  if (GID == 0) {
    // Create chare proxy and register entry methods
    comm_proxy = new charm::chare_proxy<Comm>();
    comm_proxy->add_entry_method<&entry_init>();
    comm_proxy->add_entry_method<&entry_init_done>();
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
  }
  BARRIER_LOCAL;

  barrier();

  // Execute only on PE 0
  if (pe == 0) {
    constexpr int n_params = 4;
#ifdef SM_LEVEL
    __shared__ size_t params[n_params];
#endif

    if (GID == 0) {
#ifndef SM_LEVEL
      params = new size_t[n_params];
#endif

      // Default parameters
      params[0] = 1;
      params[1] = 1048576;
      params[2] = 1000;
      params[3] = 10;

      // Process command line arguments
      if (argc >= 2) params[0] = charm::device_atoi(argv[1], argvs[1]);
      if (argc >= 3) params[1] = charm::device_atoi(argv[2], argvs[2]);
      if (argc >= 4) params[2] = charm::device_atoi(argv[3], argvs[3]);
      if (argc >= 5) params[3] = charm::device_atoi(argv[4], argvs[4]);

      printf("Pingpong\n");
      printf("PEs: %d\n", charm::n_pes());
      printf("Size: %llu - %llu\n", params[0], params[1]);
      printf("Iterations: %d (Warmup: %d)\n", (int)params[2], (int)params[3]);
    }
    BARRIER_LOCAL;

    comm_proxy->invoke_all(0, params, sizeof(size_t) * n_params);
  }
}

__device__ void Comm::init(void* arg) {
  if (GID == 0) {
    size_t* params = (size_t*)arg;
    int param_idx = 0;

    init_cnt = 0;
    min_size = params[param_idx++];
    max_size = params[param_idx++];
    cur_size = min_size;
    n_iters = static_cast<int>(params[param_idx++]);
    warmup = static_cast<int>(params[param_idx++]);
    iter = 0;
    index = charm::chare::i;
    peer = (index == 0) ? 1 : 0;
#ifdef USER_MSG
    comm_proxy->alloc_msg(msg, peer, max_size);
#else
    data = new char[max_size];
#endif

#ifdef MEASURE_INVOKE
    invoke_time = 0;
#endif
    printf("Chare %d init on PE %d\n", index, charm::my_pe());
  }
  BARRIER_LOCAL;

  comm_proxy->invoke(0, 1);
}

__device__ void Comm::init_done(void* arg) {
  if (GID == 0) {
    init_cnt++;
  }
  BARRIER_LOCAL;

  if (init_cnt == 2) {
    if (GID == 0) {
      printf("Init done\n");
    }
    send();
  }
}

__device__ void Comm::send() {
  if (index == 0) {
    // Start iteration, only measure time on chare 0
    if (GID == 0) {
      if (iter == warmup) {
        start_tp = cuda::std::chrono::system_clock::now();
      }
#ifdef DEBUG
      printf("Index %d iter %d sending size %lu\n", index, iter, cur_size);
#endif
#ifdef MEASURE_INVOKE
      if (iter >= warmup) {
        invoke_start_tp = cuda::std::chrono::system_clock::now();
      }
#endif
    }
    BARRIER_LOCAL;
#ifdef USER_MSG
    comm_proxy->invoke(peer, 2, msg, cur_size);
#else
    comm_proxy->invoke(peer, 2, data, cur_size);
#endif
#ifdef MEASURE_INVOKE
    if (GID == 0) {
      if (iter >= warmup) {
        invoke_end_tp = cuda::std::chrono::system_clock::now();
        cuda::std::chrono::duration<double> diff = invoke_end_tp - invoke_start_tp;
        invoke_time += diff.count();
      }
      if (iter == n_iters + warmup - 1) {
        printf("Time per invoke: %.3lf us\n", invoke_time / n_iters * 1000000);
        invoke_time = 0;
      }
    }
    BARRIER_LOCAL;
#endif
  } else {
    // End iteration
#ifdef DEBUG
    if (GID == 0) {
      printf("Index %d iter %d sending size %lu\n", index, iter, cur_size);
    }
    BARRIER_LOCAL;
#endif
#ifdef USER_MSG
    comm_proxy->invoke(peer, 2, msg, cur_size);
#else
    comm_proxy->invoke(peer, 2, data, cur_size);
#endif
    if (GID == 0) {
      if (++iter == n_iters + warmup) {
        cur_size *= 2;
        iter = 0;
      }
    }
    BARRIER_LOCAL;
  }
}

__device__ void Comm::recv(void* arg) {
#ifdef SM_LEVEL
  __shared__ bool end;
#endif
  if (GID == 0) {
    end = false;
  }
  BARRIER_LOCAL;

  if (GID == 0) {
    if (index == 0) {
#ifdef DEBUG
      printf("Index %d iter %d received size %lu\n", index, iter, cur_size);
#endif
      if (++iter == n_iters + warmup) {
        end_tp = cuda::std::chrono::system_clock::now();
        cuda::std::chrono::duration<double> diff = end_tp - start_tp;
        printf("Size %llu took %.3lf us\n", cur_size, (diff.count() / 2 / n_iters) * 1000000);
        cur_size *= 2;
        iter = 0;
        if (cur_size > max_size) {
          end = true;
        }
      }
    } else {
#ifdef DEBUG
      printf("Index %d iter %d received size %lu\n", index, iter, cur_size);
#endif
    }
  }
  BARRIER_LOCAL;

  if (end) {
    charm::end();
  } else {
    send();
  }
}
