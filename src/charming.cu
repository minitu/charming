#include <stdio.h>
#include <string.h>
#include <cuda.h>
#ifdef CHARMING_USE_MPI
#include <mpi.h>
#endif
#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "util.h"

// Maximum number of chare types
#define CHARE_TYPE_CNT_MAX 1024

using namespace charm;

cudaStream_t stream;

__constant__ int c_my_pe;
__constant__ int c_n_pes;
__constant__ int c_my_pe_node;
__constant__ int c_n_pes_node;
__constant__ int c_n_nodes;

__device__ chare_proxy_base* chare_proxies[CHARE_TYPE_CNT_MAX];
__device__ int chare_proxy_cnt;

int main(int argc, char* argv[]) {
#ifdef CHARMING_USE_MPI
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
#else
  nvshmem_init();
#endif // CHARMING_USE_MPI
  int h_my_pe = nvshmem_my_pe();
  int h_n_pes = nvshmem_n_pes();
  int h_my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  int h_n_pes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int h_n_nodes = h_n_pes / h_n_pes_node;

  // Initialize CUDA and create stream
  // Round-robin mapping of processes to GPUs
  int n_devices = 0;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    if (h_my_pe == 0) {
      PERROR("Need at least 1 GPU but detected %d GPUs\n", n_devices);
    }
    return -1;
  }
  cudaSetDevice(h_my_pe_node % n_devices);
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

  // Transfer constants
  cudaMemcpyToSymbolAsync(c_my_pe, &h_my_pe, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_pes, &h_n_pes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_my_pe_node, &h_my_pe_node, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_pes_node, &h_n_pes_node, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_nodes, &h_n_nodes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cuda_check_error();

  // Change device limits
  size_t stack_size, heap_size;
  constexpr size_t new_stack_size = 16384;
  cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size);
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  //constexpr size_t new_heap_size = 8589934592; // Set max heap size to 8GB
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size);
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

  // Print configuration and launch scheduler
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_blocks_sm;
  int n_sms = prop.multiProcessorCount;
  int max_threads_tb = prop.maxThreadsPerBlock;
  dim3 grid_dim = dim3(n_sms);
  dim3 block_dim = dim3(max_threads_tb);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_sm, (const void*)scheduler,
      block_dim.x*block_dim.y*block_dim.z, 0);
  cuda_check_error();
  if (h_my_pe == 0) {
    PINFO("Initiating CharminG\n");
    PINFO("PEs: %d, Nodes: %d\n", h_n_pes, h_n_nodes);
    PINFO("Thread grid: %d x %d x %d, Thread block: %d x %d x %d\n",
        grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);
    PINFO("Stack size: %llu Bytes, Heap size: %llu Bytes, Clock rate: %.2lf GHz\n",
        stack_size, heap_size, (double)prop.clockRate / 1e6);
    PINFO("Max active TBs per SM: %d, Number of SMs: %d\n", max_blocks_sm, n_sms);
  }

  // Initialize communication module
  comm_init_host(h_n_pes);
  nvshmemx_barrier_all_on_stream(stream);

  // Launch scheduler kernel
  void* kargs[] = { &argc, &d_argv, &d_argvs };
  nvshmemx_collective_launch((const void*)scheduler, grid_dim, block_dim, kargs, 0, stream);
  cudaStreamSynchronize(stream);
  cuda_check_error();
  nvshmemx_barrier_all_on_stream(stream);

  // Cleanup
  comm_fini_host(h_n_pes);
  cudaStreamDestroy(stream);
  nvshmem_finalize();
#ifdef CHARMING_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

__device__ void charm::end() {
  // TODO: Check if begin_terminate message has already been sent from this PE
  send_begin_term_msg(0);
}

__device__ int charm::my_pe() { return c_my_pe; }
__device__ int charm::n_pes() { return c_n_pes; }
__device__ int charm::my_pe_node() { return c_my_pe_node; }
__device__ int charm::n_pes_node() { return c_n_pes_node; }
__device__ int charm::n_nodes() { return c_n_nodes; }

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
