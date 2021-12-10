#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "message.h"
#include "scheduler.h"
#include "composite.h"
#include "ringbuf.h"
#include "util.h"

// Maximum number of chare types
#define CHARE_TYPE_CNT_MAX 1024

using namespace charm;

__constant__ int c_my_pe;
__constant__ int c_n_pes;

__device__ ringbuf_t* mbuf;
__device__ size_t mbuf_size;
__device__ uint64_t* send_status;
__device__ uint64_t* recv_composite;
__device__ uint64_t* send_composite;
__device__ size_t* send_status_idx;
__device__ size_t* recv_composite_idx;
__device__ composite_t* heap_buf;
__device__ size_t heap_buf_size;

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
  int h_my_pe = nvshmem_my_pe();
  int h_n_pes = nvshmem_n_pes();

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

  // Allocate message buffer
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  //size_t h_mbuf_size = prop.totalGlobalMem / 2;
  // TODO
  size_t h_mbuf_size = 1073741824;
  ringbuf_t* h_mbuf;
  ringbuf_t* h_mbuf_d;
  cudaMallocHost(&h_mbuf, sizeof(ringbuf_t));
  cudaMalloc(&h_mbuf_d, sizeof(ringbuf_t));
  assert(h_mbuf && h_mbuf_d);
  h_mbuf->init(h_mbuf_size);

  // Allocate data structures
  size_t h_status_size = MSG_IN_FLIGHT_MAX * h_n_pes * sizeof(uint64_t);
  uint64_t* h_send_status = (uint64_t*)nvshmem_malloc(h_status_size);
  size_t h_composite_size = MSG_IN_FLIGHT_MAX * h_n_pes * sizeof(uint64_t);
  uint64_t* h_recv_composite = (uint64_t*)nvshmem_malloc(h_composite_size);
  uint64_t* h_send_composite;
  cudaMalloc(&h_send_composite, h_composite_size);
  size_t h_idx_size = MSG_IN_FLIGHT_MAX * h_n_pes * sizeof(size_t);
  size_t* h_send_status_idx;
  size_t* h_recv_composite_idx;
  cudaMalloc(&h_send_status_idx, h_idx_size);
  cudaMalloc(&h_recv_composite_idx, h_idx_size);
  composite_t* h_heap_buf;
  size_t h_heap_buf_size = MSG_IN_FLIGHT_MAX * h_n_pes * 2 * sizeof(composite_t);
  cudaMalloc(&h_heap_buf, h_heap_buf_size);
  assert(h_send_status && h_recv_composite && h_send_status_idx && h_recv_composite_idx
      && h_heap_buf);
  cuda_check_error();

  // Synchronize all NVSHMEM PEs
  nvshmem_barrier_all();

  // Change device limits
  size_t stack_size, heap_size;
  // TODO
  //size_t new_heap_size = 8589934592; // Set max heap size to 8GB
  //cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size);
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

  // Print configuration and launch scheduler
  /* TODO
  int grid_size = prop.multiProcessorCount;
  int block_size = prop.maxThreadsPerBlock;
  */
  int grid_size = 1;
  int block_size = 1;
  if (rank == 0) {
    printf("CHARMING\nGrid size: %d\nBlock size: %d\nStack size: %llu B\n"
           "Heap size: %llu B\nClock rate: %.2lf GHz\n",
           grid_size, block_size, stack_size, heap_size,
           (double)prop.clockRate / 1e6);
  }
  cudaMemcpyToSymbolAsync(c_my_pe, &h_my_pe, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_pes, &h_n_pes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(h_mbuf_d, h_mbuf, sizeof(ringbuf_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf_d, sizeof(ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_status, &h_send_status, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_composite, &h_recv_composite, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_composite, &h_send_composite, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_status_idx, &h_send_status_idx, sizeof(size_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_composite_idx, &h_recv_composite_idx, sizeof(size_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(heap_buf, &h_heap_buf, sizeof(composite_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(heap_buf_size, &h_heap_buf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(h_send_status, 0, h_status_size, stream);
  cudaMemsetAsync(h_recv_composite, 0, h_composite_size, stream);
  cudaMemsetAsync(h_send_composite, 0, h_composite_size, stream);
  cudaMemsetAsync(h_send_status_idx, 0, h_idx_size, stream);
  cudaMemsetAsync(h_recv_composite_idx, 0, h_idx_size, stream);
  cudaMemsetAsync(h_heap_buf, 0, h_heap_buf_size, stream);

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
  nvshmem_free(h_send_status);
  nvshmem_free(h_recv_composite);
  cudaFree(h_send_composite);
  cudaFree(h_send_status_idx);
  cudaFree(h_recv_composite_idx);
  cudaFree(h_heap_buf);
  h_mbuf->fini();
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
  return c_n_pes;
}

__device__ int charm::my_pe() {
  return c_my_pe;
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
