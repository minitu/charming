#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda/std/chrono>

#define LOCAL_MAX 4
#define NO_ALLOC

typedef unsigned long long int atomic_t;

__constant__ int c_n_sms;
__device__ volatile void** envs;

__device__ __forceinline__ int find_free(volatile void** envs) {
  __shared__ volatile int idx;
  if (threadIdx.x == 0) idx = INT_MAX;
  __syncthreads();

  // Loop until a free index is found
  while (idx == INT_MAX) {
    for (int i = threadIdx.x; i < LOCAL_MAX; i += blockDim.x) {
      if (envs[i] == nullptr) {
        atomicMin_block((int*)&idx, i);
      }
      __threadfence();
    }
  }
  __syncthreads();

  return idx;
}

__device__ void send(int my_sm, int dst_sm) {
  volatile void** dst_envs = envs + LOCAL_MAX * c_n_sms * dst_sm
    + LOCAL_MAX * my_sm;
  int free_idx = find_free(dst_envs);

  if (threadIdx.x == 0) {
    void* msg = (void*)1;
#ifndef NO_ALLOC
    msg = (void*) new char[4];
    __threadfence();
#endif
    atomic_t ret = atomicCAS((atomic_t*)&dst_envs[free_idx], 0, (atomic_t)msg);
    assert(ret == 0);
  }
}

__device__ __forceinline__ void* find_msg(volatile void** envs, int& ret_idx) {
  __shared__ volatile int idx;
  __shared__ void* env;
  if (threadIdx.x == 0) {
    idx = INT_MAX;
    env = nullptr;
  }
  __syncthreads();

  // Look for a valid message (traverse once)
  for (int i = threadIdx.x; i < LOCAL_MAX * c_n_sms; i += blockDim.x) {
    if (envs[i] != nullptr) {
      atomicMin_block((int*)&idx, i);
    }
    __threadfence();
  }
  __syncthreads();
  ret_idx = idx;

  // If a message is found
  if (idx != INT_MAX && threadIdx.x == 0) {
    env = (void*)envs[idx];
    __threadfence();

    // Reset message address to zero
    atomic_t ret = atomicCAS((atomic_t*)&envs[idx],
        (atomic_t)env, 0);
    assert(ret == (atomic_t)env);
  }
  __syncthreads();

  return env;
}

__device__ bool recv(int my_sm) {
  // Look for valid message addresses
  volatile void** my_envs = envs + LOCAL_MAX * c_n_sms * my_sm;
  int msg_idx;
  void* env = find_msg(my_envs, msg_idx);

#ifndef NO_ALLOC
  if (env) {
    // Message cleanup
    if (threadIdx.x == 0) {
      delete[] (char*)env;
      __threadfence();
    }
    __syncthreads();
  }
#endif

  return env ? true : false;
}

__global__ void scheduler() {
  int my_sm = blockIdx.x;
  int peer_sm = my_sm ? 0 : 1;
  constexpr int n_iters = 1000;
  constexpr int warmup = 10;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> start_tp;
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock> end_tp;

  if (my_sm == 0 || my_sm == 1) {
    if (threadIdx.x == 0) {
      printf("SM %d starting\n", my_sm);
    }

    for (int i = 0; i < n_iters + warmup; i++) {
      if (i == warmup) {
        start_tp = cuda::std::chrono::system_clock::now();
      }

      if (my_sm == 0) {
        send(my_sm, peer_sm);

        while (!recv(my_sm)) {}
      } else {
        while (!recv(my_sm)) {}

        send(my_sm, peer_sm);
      }
    }

    end_tp = cuda::std::chrono::system_clock::now();
    cuda::std::chrono::duration<double> diff = end_tp - start_tp;
    if (threadIdx.x == 0) {
      printf("SM %d ending, %.lf us per iteration\n", my_sm, diff.count() / 2 / n_iters * 1e6);
    }
  }
}

int main() {
  int n_sms = 80;
  cudaMemcpyToSymbol(c_n_sms, &n_sms, sizeof(int));

  atomic_t* d_envs;
  size_t env_size = sizeof(void*) * LOCAL_MAX * n_sms * n_sms;
  cudaMalloc(&d_envs, env_size);
  assert(d_envs);
  cudaMemset(d_envs, 0, env_size);
  cudaMemcpyToSymbol(envs, &d_envs, sizeof(atomic_t*));

  dim3 grid_dim(n_sms);
  dim3 block_dim(512);

  scheduler<<<grid_dim, block_dim>>>();
  cudaDeviceSynchronize();

  cudaFree(d_envs);

  return 0;
}
