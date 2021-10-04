#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "charming.h"
#include "message.h"
#include "scheduler.h"
#include "msg_queue.h"
#include "ringbuf.h"
#include "util.h"

#define CHARE_TYPE_CNT_MAX 1024 // Maximum number of chare types

__device__ MsgQueueMetaShell* recv_meta_shell; // For receives
__device__ MsgQueueMetaShell* send_meta_shell; // For sends
__device__ MsgQueueShell* msg_queue_shell;
__device__ size_t msg_queue_size;
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
  int n_pes = nvshmem_n_pes();
  size_t h_msg_queue_size = (1 << 28);
  MsgQueueMetaShell* h_recv_meta_shell;
  MsgQueueMetaShell* h_send_meta_shell;
  MsgQueueShell* h_msg_queue_shell;
  cudaMallocHost(&h_recv_meta_shell, sizeof(MsgQueueMetaShell) * n_pes);
  cudaMallocHost(&h_send_meta_shell, sizeof(MsgQueueMetaShell) * n_pes);
  cudaMallocHost(&h_msg_queue_shell, sizeof(MsgQueueShell) * n_pes);
  for (int i = 0; i < n_pes; i++) {
    new (&h_recv_meta_shell[i]) MsgQueueMetaShell(h_msg_queue_size);
    new (&h_send_meta_shell[i]) MsgQueueMetaShell(h_msg_queue_size);
    new (&h_msg_queue_shell[i]) MsgQueueShell(h_msg_queue_size);
  }
  MsgQueueMetaShell* d_recv_meta_shell;
  MsgQueueMetaShell* d_send_meta_shell;
  MsgQueueShell* d_msg_queue_shell;
  cudaMalloc(&d_recv_meta_shell, sizeof(MsgQueueMetaShell) * n_pes);
  cudaMalloc(&d_send_meta_shell, sizeof(MsgQueueMetaShell) * n_pes);
  cudaMalloc(&d_msg_queue_shell, sizeof(MsgQueueShell) * n_pes);
  size_t h_mbuf_size = (1 << 28);
  spsc_ringbuf_t* h_mbuf = spsc_ringbuf_malloc(h_mbuf_size);
  cuda_check_error();
  nvshmem_barrier_all();

  // Launch scheduler
  int grid_size = 1;
  int block_size = 1;
  //cudaDeviceSetLimit(cudaLimitStackSize, 16384);
  size_t stack_size, heap_size;
  size_t new_heap_size = 8589934592; // Set heap max to 8GB
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size);
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  if (!rank) {
    printf("CHARMING\nGrid size: %d\nBlock size: %d\nStack size: %llu\n"
           "Heap size: %llu\nClock rate: %.2lf GHz\n",
           grid_size, block_size, stack_size, heap_size,
           (double)prop.clockRate / 1e6);
  }
  //void* scheduler_args[4] = { &rbuf, &rbuf_size, &mbuf, &mbuf_size };
  cudaMemcpyAsync(d_recv_meta_shell, h_recv_meta_shell,
      sizeof(MsgQueueMetaShell) * n_pes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_send_meta_shell, h_send_meta_shell,
      sizeof(MsgQueueMetaShell) * n_pes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_msg_queue_shell, h_msg_queue_shell,
      sizeof(MsgQueueShell) * n_pes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_meta_shell, &d_recv_meta_shell, sizeof(MsgQueueMetaShell*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_meta_shell, &d_send_meta_shell, sizeof(MsgQueueMetaShell*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(msg_queue_shell, &d_msg_queue_shell, sizeof(MsgQueueShell*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(msg_queue_size, &h_msg_queue_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf, sizeof(spsc_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cuda_check_error();
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
  for (int i = 0; i < n_pes; i++) {
    delete &h_recv_meta_shell[i];
    delete &h_send_meta_shell[i];
    delete &h_msg_queue_shell[i];
  }
  cudaFreeHost(h_recv_meta_shell);
  cudaFreeHost(h_send_meta_shell);
  cudaFreeHost(h_msg_queue_shell);
  spsc_ringbuf_free(h_mbuf);
  nvshmem_finalize();
  cudaStreamDestroy(stream);
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
