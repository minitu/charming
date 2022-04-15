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

// GPU constant memory
__constant__ int c_n_sms;
__constant__ int c_my_dev;
__constant__ int c_my_dev_node;
__constant__ int c_n_devs;
__constant__ int c_n_devs_node;
__constant__ int c_n_pes;
__constant__ int c_n_pes_node;
__constant__ int c_n_nodes;

// GPU global memory
__device__ chare_proxy_base* chare_proxies[CHARE_TYPE_CNT_MAX];
__device__ int chare_proxy_cnt;

// GPU shared memory
extern __shared__ uint64_t s_mem[];

int main(int argc, char* argv[]) {
  // Initialize NVSHMEM (and MPI if needed)
#ifdef CHARMING_USE_MPI
  MPI_Init(&argc, &argv);
  int world_size;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  nvshmemx_init_attr_t attr;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#else
  nvshmem_init();
#endif

  // Execution environment
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads_tb = prop.maxThreadsPerBlock;
  int h_n_sms = prop.multiProcessorCount;
  int h_my_dev = nvshmem_my_pe();
  int h_my_dev_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  int h_n_devs = nvshmem_n_pes();
  int h_n_devs_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int h_n_pes = h_n_devs * h_n_sms;
  int h_n_pes_node = h_n_devs_node * h_n_sms;
  int h_n_nodes = h_n_devs / h_n_devs_node;

  // Initialize CUDA and create stream
  // Round-robin mapping of processes to GPUs
  int n_devices = 0;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    if (h_my_dev == 0) {
      PERROR("Need at least 1 GPU but detected %d GPUs\n", n_devices);
    }
    return -1;
  }
  cudaSetDevice(h_my_dev_node % n_devices);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cuda_check_error();

  // Transfer command line arguments to GPU
  char* m_args; // Contains the actual arguments consecutively
  char** m_argv; // Contains pointers to the arguments
  size_t* m_argvs; // Contains sizes of the arguments
  size_t argvs_total = 0; // Sum of all argument sizes

  // Figure out size of each argument and total size
  cudaMallocManaged(&m_argvs, sizeof(size_t) * argc);
  for (int i = 0; i < argc; i++) {
    m_argvs[i] = strlen(argv[i]);
    argvs_total += m_argvs[i] + 1; // Include NULL character
  }

  // Allocate memory for actual arguments
  cudaMallocManaged(&m_args, argvs_total);

  // Copy arguments into managed memory and store their addresses
  cudaMallocManaged(&m_argv, sizeof(char*) * argc);
  char* cur_arg = m_args;
  for (int i = 0; i < argc; i++) {
    strcpy(cur_arg, argv[i]);
    m_argv[i] = cur_arg;

    cur_arg += m_argvs[i] + 1; // Include NULL character
  }

  // Transfer execution environment constants
  cudaMemcpyToSymbolAsync(c_n_sms, &h_n_sms, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_my_dev, &h_my_dev, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_my_dev_node, &h_my_dev_node, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_devs, &h_n_devs, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_devs_node, &h_n_devs_node, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_pes, &h_n_pes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_pes_node, &h_n_pes_node, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_nodes, &h_n_nodes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cuda_check_error();

  // Change device limits
  size_t stack_size, heap_size;
  size_t smem_size = SMEM_CNT_MAX * sizeof(uint64_t) + 128;
  constexpr size_t new_stack_size = 16384;
  cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size);
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  //constexpr size_t new_heap_size = 8589934592; // Set max heap size to 8GB
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size);
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

  // Print configuration and launch scheduler
  dim3 grid_dim = dim3(h_n_sms);
  //dim3 block_dim = dim3(max_threads_tb);
  dim3 block_dim = dim3(512);
  int max_blocks_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_sm, (const void*)scheduler,
      block_dim.x*block_dim.y*block_dim.z, 0);
  cuda_check_error();
  if (h_my_dev == 0) {
    PINFO("Initiating CharminG\n");
    PINFO("PEs: %d, Nodes: %d\n", h_n_pes, h_n_nodes);
    PINFO("Thread grid: %d x %d x %d, Thread block: %d x %d x %d\n",
        grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);
    PINFO("Stack size: %llu Bytes, Heap size: %llu Bytes\n", stack_size, heap_size);
    PINFO("Shared memory size: %llu Bytes, Clock rate: %.2lf GHz\n",
        smem_size, (double)prop.clockRate / 1e6);
    PINFO("Max active TBs per SM: %d, Number of SMs: %d\n", max_blocks_sm, h_n_sms);
  }

  // Initialize communication module
  comm_init_host(h_n_pes, h_n_sms);
  nvshmemx_barrier_all_on_stream(stream);

  // Launch scheduler kernel
  void* kargs[] = { &argc, &m_argv, &m_argvs };
  nvshmemx_collective_launch((const void*)scheduler, grid_dim, block_dim, kargs,
      smem_size, stream);
  printf("After collective launch\n");
  cudaStreamSynchronize(stream);
  cuda_check_error();
  nvshmemx_barrier_all_on_stream(stream);
  printf("After nvshmemx barrier\n");

  // Cleanup
  comm_fini_host(h_n_pes, h_n_sms);
  cudaStreamDestroy(stream);
  printf("After cudaStreamDestroy\n");
  nvshmem_finalize();
  printf("After nvshmemx_finalize\n");
#ifdef CHARMING_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

__device__ void charm::end() {
  // TODO: Check if begin_terminate message has already been sent from this PE
  send_begin_term_msg(0);
}

__device__ int charm::my_pe() { return s_mem[3]; }
__device__ int charm::n_pes() { return c_n_pes; }
__device__ int charm::my_pe_node() { return s_mem[4]; }
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
