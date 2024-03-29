#include <stdio.h>
#include <string>
#include <string.h>
#include <cuda.h>
#ifdef CHARMING_USE_MPI
#include <mpi.h>
#endif
#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "common.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "util.h"

#define NVSHMEM_MAX_SIZE 2147483648

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

#ifdef SM_LEVEL
// GPU constant memory
__constant__ int c_n_clusters_dev;
__constant__ int c_cluster_size;
__constant__ int c_n_pes_cluster;
__constant__ int c_n_ces_cluster;
__constant__ int c_n_ces;
__constant__ int c_n_ces_node;

// GPU shared memory
extern __shared__ uint64_t s_mem[];
#endif

// GPU global memory
__device__ __managed__ chare_proxy_table* proxy_tables;

int main(int argc, char* argv[]) {
  // Increase maximum NVSHMEM memory size
  std::string env_str = "NVSHMEM_SYMMETRIC_SIZE=";
  env_str += std::to_string(NVSHMEM_MAX_SIZE);
  putenv(const_cast<char*>(env_str.c_str()));

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

  // Execute application's host main function
  charm::main_host(argc, argv);

  // Execution environment
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  //int max_threads_tb = prop.maxThreadsPerBlock;
  int h_n_sms = prop.multiProcessorCount;
  int h_my_dev = nvshmem_my_pe();
  int h_my_dev_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  int h_n_devs = nvshmem_n_pes();
  int h_n_devs_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  int h_n_nodes = h_n_devs / h_n_devs_node;
#ifdef SM_LEVEL
  int h_n_clusters_dev = 1;
  int h_cluster_size = h_n_sms / h_n_clusters_dev;
  int h_n_ces_cluster = 1;
  int h_n_pes_cluster = h_cluster_size - h_n_ces_cluster;
  int h_n_pes = h_n_devs * h_n_clusters_dev * h_n_pes_cluster;
  int h_n_ces = h_n_devs * h_n_clusters_dev * h_n_ces_cluster;
  int h_n_ces_node = h_n_ces / h_n_nodes;

  // Check if number of PE clusters is valid
  if (h_n_sms % h_n_clusters_dev != 0) {
    if (h_my_dev == 0) {
      PERROR("Number of PE clusters must be a factor of the number of SMs\n");
    }
    return -1;
  }
#else
  int h_n_pes = h_n_devs;
#endif
  int h_n_pes_node = h_n_pes / h_n_nodes;

  // Check for necessary CUDA functionalities
  if (!prop.cooperativeLaunch) {
    if (h_my_dev == 0) {
      PERROR("Need support for CUDA Cooperative Groups\n");
    }
    return -1;
  }
  if (!prop.managedMemory) {
    if (h_my_dev == 0) {
      PERROR("Need support for CUDA Unified Memory\n");
    }
    return -1;
  }

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

  // Create chare proxy tables
#ifdef SM_LEVEL
  cudaMallocManaged(&proxy_tables, sizeof(chare_proxy_table) * h_n_sms);
  for (int i = 0; i < h_n_sms; i++) {
    new (&proxy_tables[i]) chare_proxy_table();
  }
#else
  cudaMallocManaged(&proxy_tables, sizeof(chare_proxy_table));
  new (&proxy_tables[0]) chare_proxy_table();
#endif

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
#ifdef SM_LEVEL
  cudaMemcpyToSymbolAsync(c_n_clusters_dev, &h_n_clusters_dev, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_cluster_size, &h_cluster_size, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_pes_cluster, &h_n_pes_cluster, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_ces_cluster, &h_n_ces_cluster, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_ces, &h_n_ces, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_ces_node, &h_n_ces_node, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
#endif
  cuda_check_error();

  // Change device limits
  size_t stack_size, heap_size;
#ifdef SM_LEVEL
  size_t smem_size = SMEM_CNT_MAX * sizeof(uint64_t) + 128;
#else
  size_t smem_size = 0;
#endif
  constexpr size_t new_stack_size = 16384;
  cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size);
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  constexpr size_t new_heap_size = 4294967296;
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size);
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cuda_check_error();

  // Print configuration
  dim3 grid_dim = dim3(h_n_sms);
  //dim3 block_dim = dim3(max_threads_tb);
  dim3 block_dim = dim3(512);
  int max_blocks_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_sm, (const void*)scheduler,
      block_dim.x*block_dim.y*block_dim.z, 0);
  cuda_check_error();
  if (h_my_dev == 0) {
    PINFO("Initiating CharminG\n");
    PINFO("PEs: %d, GPU devices: %d, Nodes: %d\n", h_n_pes, h_n_devs, h_n_nodes);
#ifdef SM_LEVEL
    PINFO("PE Clusters per device: %d (%d PEs, %d CEs per cluster)\n",
        h_n_clusters_dev, h_n_pes_cluster, h_n_ces_cluster);
#endif
    PINFO("Thread grid: %d x %d x %d, Thread block: %d x %d x %d\n",
        grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);
    PINFO("Stack size: %llu Bytes, Heap size: %llu Bytes\n", stack_size, heap_size);
    PINFO("Shared memory size: %llu Bytes, Clock rate: %.2lf GHz\n",
        smem_size, (double)prop.clockRate / 1e6);
    PINFO("Max active TBs per SM: %d, Number of SMs: %d\n", max_blocks_sm, h_n_sms);
  }

  // Initialize communication module
#ifdef SM_LEVEL
  comm_init_host(h_n_sms, h_n_pes, h_n_ces, h_n_clusters_dev,
      h_n_pes_cluster, h_n_ces_cluster);
#else
  comm_init_host(h_n_pes);
#endif
  nvshmemx_barrier_all_on_stream(stream);

  // Launch scheduler kernel
  void* kargs[] = { &argc, &m_argv, &m_argvs };
  nvshmemx_collective_launch((const void*)scheduler, grid_dim, block_dim, kargs,
      smem_size, stream);
  cudaStreamSynchronize(stream);
  cuda_check_error();
  nvshmemx_barrier_all_on_stream(stream);

  if (h_my_dev == 0) {
    PINFO("Exiting CharminG\n");
  }

  // Cleanup
  comm_fini_host();
  cudaFree(proxy_tables);
  cudaFree(m_argvs);
  cudaFree(m_args);
  cudaFree(m_argv);
  cudaStreamDestroy(stream);
  nvshmem_finalize();
#ifdef CHARMING_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

__device__ void charm::end() {
  send_begin_term_msg();
}

#ifdef SM_LEVEL
__device__ void charm::abort() {
  // Abort currently running PE
  if (threadIdx.x == 0) {
    comm* c = (comm*)(s_mem + SMEM_CNT_MAX);
    c->do_term_flag = true;
  }
  __syncthreads();
}
#endif

__device__ int charm::my_pe() {
#ifdef SM_LEVEL
  return s_mem[s_idx::my_pe];
#else
  return c_my_dev;
#endif
}
__device__ int charm::n_pes() { return c_n_pes; }
__device__ int charm::n_pes_node() { return c_n_pes_node; }
__device__ int charm::n_nodes() { return c_n_nodes; }

__device__ void charm::barrier() { scheduler_barrier(); }
__device__ void charm::barrier_local() { scheduler_barrier_local(); }

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
