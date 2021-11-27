#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "message.h"
#include "scheduler.h"
#include "ringbuf.h"
#include "util.h"

#define CHARE_TYPE_CNT_MAX 1024 // Maximum number of chare types

using namespace charm;

__device__ spsc_ringbuf_t* mbuf;
__device__ size_t mbuf_size;

__device__ chare_proxy_base* chare_proxies[CHARE_TYPE_CNT_MAX];
__device__ int chare_proxy_cnt;

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);
  int world_size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Initialize NVSHMEM
  nvshmemx_init_attr_t attr;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  int n_pes = nvshmem_n_pes();

  // Initialize CUDA
  // FIXME: Always mapped to first device
  int n_devices = 0;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    if (rank == 0) {
      printf("ERROR: Need at least 1 GPU but detected %d GPUs\n", n_devices);
    }
    return -1;
  }
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cuda_check_error();

  // Transfer command line arguments to GPU
  size_t h_argvs[argc];
  size_t argvs_total = 0;
  for (int i = 0; i < argc; i++) {
    h_argvs[i] = strlen(argv[i]);
    argvs_total += h_argvs[i] + 1; // Include NULL character
  }
  size_t* d_argvs;
  cudaMalloc(&d_argvs, sizeof(size_t) * argc);
  cudaMemcpyAsync(d_argvs, h_argvs, sizeof(size_t) * argc, cudaMemcpyHostToDevice, stream);
  char* d_argvv;
  cudaMalloc(&d_argvv, argvs_total);
  char* h_argv[argc];
  h_argv[0] = d_argvv;
  cudaMemcpyAsync(h_argv[0], argv[0], h_argvs[0] + 1, cudaMemcpyHostToDevice, stream);
  for (int i = 1; i < argc; i++) {
    h_argv[i] = h_argv[i-1] + h_argvs[i-1] + 1;
    cudaMemcpyAsync(h_argv[i], argv[i], h_argvs[i] + 1, cudaMemcpyHostToDevice, stream);
  }
  char** d_argv;
  cudaMalloc(&d_argv, sizeof(char*) * argc);
  cudaMemcpyAsync(d_argv, h_argv, sizeof(char*) * argc, cudaMemcpyHostToDevice, stream);
  cuda_check_error();

  // Allocate message buffer using NVSHMEM, whose size will be
  // half of available GPU memory
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  size_t h_mbuf_size = prop.totalGlobalMem / 2;
  spsc_ringbuf_t* h_mbuf = spsc_ringbuf_malloc(h_mbuf_size);
  cuda_check_error();

  // Synchronize all NVSHMEM PEs
  nvshmem_barrier_all();

  // Change device limits
  size_t stack_size, heap_size;
  size_t new_heap_size = 8589934592; // Set max heap size to 8GB
  //cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size);
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

  // Print configuration and launch scheduler
  int grid_size = prop.multiProcessorCount;
  int block_size = prop.maxThreadsPerBlock;
  if (rank == 0) {
    printf("CHARMING\nGrid size: %d\nBlock size: %d\nStack size: %llu B\n"
           "Heap size: %llu B\nClock rate: %.2lf GHz\n",
           grid_size, block_size, stack_size, heap_size,
           (double)prop.clockRate / 1e6);
  }
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf, sizeof(spsc_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cuda_check_error();
  /* This doesn't support CUDA dynamic parallelism, will it be a problem?
  void* scheduler_args[4] = { &rbuf, &rbuf_size, &mbuf, &mbuf_size };
  nvshmemx_collective_launch((const void*)scheduler, grid_size, block_size,
      //scheduler_args, 0, stream);
      nullptr, 0, stream);
  */
  scheduler<<<grid_size, block_size, 0, stream>>>(argc, d_argv, d_argvs);
  cudaStreamSynchronize(stream);
  cuda_check_error();

  //nvshmemx_barrier_all_on_stream(stream); // Hangs
  nvshmem_barrier_all();

  // Cleanup
  spsc_ringbuf_free(h_mbuf);
  cudaStreamDestroy(stream);
  nvshmem_finalize();
  MPI_Finalize();

  return 0;
}

__device__ void charm::end() {
  // TODO: Check if begin_terminate message has already been sent from this PE
  send_begin_term_msg(0);
}

__device__ int charm::n_pes() {
  return nvshmem_n_pes();
}

__device__ int charm::my_pe() {
  return nvshmem_my_pe();
}

__device__ int charm::device_atoi(const char* str, int strlen) {
  int tmp = 0;
  for (int i = 0; i < strlen; i++) {
    int multiplier = 1;
    for (int j = 0; j < strlen - i - 1; j++) {
      multiplier *= 10;
    }
    tmp += (str[i] - 48) * multiplier;
  }
  return tmp;
}
