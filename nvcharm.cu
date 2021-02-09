#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "Message.h"

namespace cg = cooperative_groups;

__device__ uint get_smid(void) {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

__global__ void scheduler(int* sm_ids) {
  if (threadIdx.x == 0) {
    // Store SM ID
    sm_ids[blockIdx.x] = get_smid();

    Message* msg;
    msg = (Message*)malloc(sizeof(Message));
    msg->ep = 1;
    msg->data = msg;
  }
}

int main(int argc, char* argv[]) {
  // Print GPU device properties
  int device = 0;
  cudaSetDevice(device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("* GPU properties\n"
      "Name: %s\nCompute capability: %d.%d\nSMs: %d\n"
      "Max threads per SM: %d\nKernel runtime limit: %d\n"
      "Managed memory support: %d\nCooperative kernel support: %d\n\n",
      prop.name, prop.major, prop.minor, prop.multiProcessorCount,
      prop.maxThreadsPerMultiProcessor, prop.kernelExecTimeoutEnabled,
      prop.managedMemory, prop.cooperativeLaunch);

  if (!prop.managedMemory) {
    fprintf(stderr, "Managed memory support required\n");
    exit(1);
  }

  // Obtain kernel block and grid sizes
  int block_size = 1;
  int grid_size = prop.multiProcessorCount;
  if (argc > 1) block_size = atoi(argv[1]);
  if (argc > 2) grid_size = atoi(argv[2]);
  printf("* Test properties\n"
      "Block size: %d\nGrid size: %d\n\n", block_size, grid_size);

  // Allocate memory for SM IDs
  int* sm_ids;
  cudaMallocManaged(&sm_ids, grid_size * sizeof(int));

  // Run kernel
  scheduler<<<grid_size, block_size>>>(sm_ids);
  cudaDeviceSynchronize();

  // Print block to SM mappings
  for (int i = 0; i < grid_size; i++) {
    printf("Block %d -> SM %d\n", i, sm_ids[i]);
  }
  printf("\n");

  cudaFree(sm_ids);

  return 0;
}
