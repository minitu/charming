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
    peer = (index == 0) ? 1 : 0;
#ifdef MEASURE_INVOKE
    invoke_time = 0;
#endif
    printf("Chare %d init\n", index);
  }
  __syncthreads();

  comm_proxy->invoke(0, 1);
}

__device__ void Comm::init_done(void* arg) {
  if (++init_cnt == 2) {
    if (threadIdx.x == 0) {
      printf("Init done\n");
    }

    send();
  }
}

__device__ void Comm::send() {
  if (index == 0) {
    // Start iteration, only measure time on chare 0
    if (threadIdx.x == 0) {
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
#ifdef USER_MSG
    comm_proxy->invoke(peer, 2, msg, cur_size);
#else
    comm_proxy->invoke(peer, 2, data, cur_size);
#endif
    __syncthreads();
#ifdef MEASURE_INVOKE
    if (threadIdx.x == 0) {
      if (iter >= warmup) {
        invoke_end_tp = cuda::std::chrono::system_clock::now();
      }
      cuda::std::chrono::duration<double> diff = invoke_end_tp - invoke_start_tp;
      invoke_time += diff.count();
      if (iter == n_iters + warmup - 1) {
        printf("Time per invoke: %.3lf us\n", invoke_time / n_iters * 1000000);
      }
    }
#endif
  } else {
    // End iteration
#ifdef DEBUG
    if (threadIdx.x == 0) {
      printf("Index %d iter %d sending size %lu\n", index, iter, cur_size);
    }
#endif
#ifdef USER_MSG
    comm_proxy->invoke(peer, 2, msg, cur_size);
#else
    comm_proxy->invoke(peer, 2, data, cur_size);
#endif
    __syncthreads();
    if (threadIdx.x == 0) {
      if (++iter == n_iters + warmup) {
        cur_size *= 2;
        iter = 0;
      }
    }
  }
}

__device__ void Comm::recv(void* arg) {
  __shared__ bool end;
  if (threadIdx.x == 0) {
    end = false;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
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
          charm::end();
          end = true;
        }
      }
    } else {
#ifdef DEBUG
      printf("Index %d iter %d received size %lu\n", index, iter, cur_size);
#endif
    }
  }
  __syncthreads();

  if (end) return;

  send();
}

// Main
__device__ void charm::main(int argc, char** argv, size_t* argvs) {
  __shared__ bool end;
  constexpr int n_params = 4;
  size_t params[n_params] = {1, 65536, 100, 10};

  if (threadIdx.x == 0) {
    end = false;

    if (charm::n_pes() != 2) {
      printf("Need exactly 2 PEs!\n");
      end = true;
      charm::end();
    }

    // Process command line arguments
    if (argc >= 2) params[0] = charm::device_atoi(argv[1], argvs[1]);
    if (argc >= 3) params[1] = charm::device_atoi(argv[2], argvs[2]);
    if (argc >= 4) params[2] = charm::device_atoi(argv[3], argvs[3]);
    if (argc >= 5) params[3] = charm::device_atoi(argv[4], argvs[4]);

    printf("Pingpong\n");
    printf("Size: %llu - %llu\n", params[0], params[1]);
    printf("Iterations: %d (Warmup: %d)\n", (int)params[2], (int)params[3]);
  }
  __syncthreads();

  if (end) return;

  Comm comm;
  comm_proxy->create(comm, 2);
  __syncthreads();

  comm_proxy->invoke_all(0, params, sizeof(size_t) * n_params);
  __syncthreads();
}
