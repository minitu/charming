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
#include "scheduler.h"
#include "composite.h"
#include "ringbuf.h"
#include "util.h"

// Maximum number of chare types
#define CHARE_TYPE_CNT_MAX 1024

using namespace charm;

__constant__ int c_my_pe;
__constant__ int c_n_pes;
__constant__ int c_my_pe_node;
__constant__ int c_n_pes_node;
__constant__ int c_n_nodes;

__device__ ringbuf_t* mbuf;
__device__ size_t mbuf_size;
__device__ uint64_t* send_status;
__device__ uint64_t* recv_remote_comp;
__device__ compbuf_t* recv_local_comp;
__device__ uint64_t* send_comp;
__device__ size_t* send_status_idx;
__device__ size_t* recv_remote_comp_idx;
__device__ composite_t* heap_buf;
__device__ size_t heap_buf_size;

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

  // Initialize CUDA
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
  size_t h_status_size = REMOTE_MSG_COUNT_MAX * h_n_pes * sizeof(uint64_t);
  uint64_t* h_send_status = (uint64_t*)nvshmem_malloc(h_status_size);
  size_t h_remote_comp_size = REMOTE_MSG_COUNT_MAX * h_n_pes * sizeof(uint64_t);
  uint64_t* h_recv_remote_comp = (uint64_t*)nvshmem_malloc(h_remote_comp_size);
  compbuf_t* h_recv_local_comp;
  compbuf_t* h_recv_local_comp_d;
  cudaMallocHost(&h_recv_local_comp, sizeof(compbuf_t));
  cudaMalloc(&h_recv_local_comp_d, sizeof(compbuf_t));
  assert(h_recv_local_comp && h_recv_local_comp_d);
  h_recv_local_comp->init(LOCAL_MSG_COUNT_MAX);
  uint64_t* h_send_comp;
  cudaMalloc(&h_send_comp, h_remote_comp_size);
  size_t* h_send_status_idx;
  size_t* h_recv_remote_comp_idx;
  size_t h_idx_size = REMOTE_MSG_COUNT_MAX * h_n_pes * sizeof(size_t);
  cudaMalloc(&h_send_status_idx, h_idx_size);
  cudaMalloc(&h_recv_remote_comp_idx, h_idx_size);
  composite_t* h_heap_buf;
  size_t h_heap_buf_size = REMOTE_MSG_COUNT_MAX * h_n_pes * 2 * sizeof(composite_t);
  cudaMalloc(&h_heap_buf, h_heap_buf_size);
  assert(h_send_status && h_recv_remote_comp && h_send_comp && h_send_status_idx
      && h_recv_remote_comp_idx && h_heap_buf);
  cuda_check_error();

  // Synchronize all NVSHMEM PEs
  nvshmem_barrier_all();

  // Change device limits
  size_t stack_size, heap_size;
  constexpr size_t new_stack_size = 16384;
  cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size);
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  //constexpr size_t new_heap_size = 8589934592; // Set max heap size to 8GB
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, new_heap_size);
  cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

  // Print configuration and launch scheduler
  /* TODO
  dim3 grid_dim = dim3(prop.multiProcessorCount);
  dim3 block_dim = dim3(prop.maxThreadsPerBlock);
  */
  dim3 grid_dim = dim3(1);
  dim3 block_dim = dim3(1);
  if (h_my_pe == 0) {
    PINFO("Initiating CharminG\n");
    PINFO("PEs: %d, Nodes: %d\n", h_n_pes, h_n_nodes);
    PINFO("Thread grid: %d x %d x %d, Thread block: %d x %d x %d\n",
        grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);
    PINFO("Stack size: %llu Bytes, Heap size: %llu Bytes, Clock rate: %.2lf GHz\n",
        stack_size, heap_size, (double)prop.clockRate / 1e6);
  }

  cudaMemcpyToSymbolAsync(c_my_pe, &h_my_pe, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_pes, &h_n_pes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_my_pe_node, &h_my_pe_node, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_pes_node, &h_n_pes_node, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(c_n_nodes, &h_n_nodes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);

  cudaMemcpyAsync(h_mbuf_d, h_mbuf, sizeof(ringbuf_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf_d, sizeof(ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);

  cudaMemcpyToSymbolAsync(send_status, &h_send_status, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_remote_comp, &h_recv_remote_comp, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(h_recv_local_comp_d, h_recv_local_comp, sizeof(compbuf_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_local_comp, &h_recv_local_comp_d, sizeof(compbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_comp, &h_send_comp, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_status_idx, &h_send_status_idx, sizeof(size_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_remote_comp_idx, &h_recv_remote_comp_idx, sizeof(size_t*), 0, cudaMemcpyHostToDevice, stream);

  cudaMemsetAsync(h_send_status, 0, h_status_size, stream);
  cudaMemsetAsync(h_recv_remote_comp, 0, h_remote_comp_size, stream);
  cudaMemsetAsync(h_send_comp, 0, h_remote_comp_size, stream);
  cudaMemsetAsync(h_send_status_idx, 0, h_idx_size, stream);
  cudaMemsetAsync(h_recv_remote_comp_idx, 0, h_idx_size, stream);

  cudaMemcpyToSymbolAsync(heap_buf, &h_heap_buf, sizeof(composite_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(heap_buf_size, &h_heap_buf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(h_heap_buf, 0, h_heap_buf_size, stream);

  cuda_check_error();
  nvshmem_barrier_all();

  /* This doesn't support CUDA dynamic parallelism, will it be a problem?
  void* scheduler_args[4] = { &rbuf, &rbuf_size, &mbuf, &mbuf_size };
  nvshmemx_collective_launch((const void*)scheduler, grid_dim, block_dim,
      //scheduler_args, 0, stream);
      nullptr, 0, stream);
  */
  scheduler<<<grid_dim, block_dim, 0, stream>>>(argc, d_argv, d_argvs);
  cudaStreamSynchronize(stream);
  cuda_check_error();

  //nvshmemx_barrier_all_on_stream(stream); // Hangs
  nvshmem_barrier_all();

  // Cleanup
  nvshmem_free(h_send_status);
  nvshmem_free(h_recv_remote_comp);
  cudaFree(h_send_comp);
  cudaFree(h_send_status_idx);
  cudaFree(h_recv_remote_comp_idx);
  cudaFree(h_heap_buf);
  h_mbuf->fini();
  cudaFreeHost(h_mbuf);
  cudaFree(h_mbuf_d);
  h_recv_local_comp->fini();
  cudaFreeHost(h_recv_local_comp);
  cudaFree(h_recv_local_comp_d);
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
