#include <assert.h>
#include <cuda_runtime.h>

#define LOCAL_MAX 4

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
    void* msg = (void*) new char[4];
    __threadfence();
    atomic_t ret = atomicCAS((atomic_t*)&dst_envs[free_idx], 0, (atomic_t)msg);
    assert(ret == 0);
  }
}

__device__ __forceinline__ void find_msg() {
  // TODO
}

__device__ bool recv() {
  // TODO
}

__global__ void scheduler() {
  constexpr int n_iters = 0;
  constexpr int warmup = 0;

  for (int i = 0; i < n_iters; i++) {
    if (blockIdx.x == 0) {
      send(0, 1);

      while (!recv()) {}
    } else {
      while (!recv()) {}

      send(1, 0);
    }
  }
}

int main() {
  int n_sms = 2;
  cudaMemcpyToSymbol(c_n_sms, &n_sms, sizeof(int));

  atomic_t* d_envs;
  cudaMalloc(&d_envs, sizeof(atomic_t) * n_sms * LOCAL_MAX);
  cudaMemcpyToSymbol(envs, &d_envs, sizeof(atomic_t*));

  dim3 grid_dim(n_sms);
  dim3 block_dim(128);

  scheduler<<<grid_dim, block_dim>>>();
  cudaDeviceSynchronize();

  cudaFree(d_envs);

  return 0;
}
