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

__device__ mpsc_ringbuf_t* rbuf;
__device__ size_t rbuf_size;
__device__ spsc_ringbuf_t* mbuf;
__device__ size_t mbuf_size;

using namespace charm;

__device__ chare_proxy_base* chare_proxies[CHARE_TYPE_CNT_MAX];
__device__ int chare_proxy_cnt;

int main(int argc, char* argv[]) {
  int rank;
  cudaStream_t stream;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Initialize NVSHMEM
  nvshmemx_init_attr_t attr;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  // Initialize CUDA
  cudaSetDevice(0);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

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

  // Allocate message queue with NVSHMEM
  size_t h_rbuf_size = (1 << 28);
  mpsc_ringbuf_t* h_rbuf = mpsc_ringbuf_malloc(h_rbuf_size);
  size_t h_mbuf_size = (1 << 28);
  spsc_ringbuf_t* h_mbuf = spsc_ringbuf_malloc(h_mbuf_size);
  nvshmem_barrier_all();

  // Launch scheduler
  int grid_size = 1;
  int block_size = 1;
  //cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  size_t stack_size;
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  if (!rank) {
    printf("CHARMING\nGrid size: %d\nBlock size: %d\nStack size: %llu\nClock rate: %.2lf GHz\n",
           grid_size, block_size, stack_size, (double)prop.clockRate / 1e6);
  }
  //void* scheduler_args[4] = { &rbuf, &rbuf_size, &mbuf, &mbuf_size };
  cudaMemcpyToSymbolAsync(rbuf, &h_rbuf, sizeof(mpsc_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(rbuf_size, &h_rbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf, sizeof(spsc_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  /* TODO: This doesn't support CUDA dynamic parallelism, will it be a problem?
  nvshmemx_collective_launch((const void*)scheduler, grid_size, block_size,
      //scheduler_args, 0, stream);
      nullptr, 0, stream);
      */
  scheduler<<<grid_size, block_size, 0, stream>>>(argc, d_argv, d_argvs);
  cuda_check_error();
  cudaStreamSynchronize(stream);
  //nvshmemx_barrier_all_on_stream(stream); // Hangs
  nvshmem_barrier_all();

  // Finalize NVSHMEM and MPI
  spsc_ringbuf_free(h_mbuf);
  mpsc_ringbuf_free(h_rbuf);
  nvshmem_finalize();
  cudaStreamDestroy(stream);
  MPI_Finalize();

  return 0;
}

__device__ void charm::exit() {
  int n_pes = nvshmem_n_pes();
  for (int pe = 0; pe < n_pes; pe++) {
    send_term_msg(pe);
  }
}
