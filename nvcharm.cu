#include <stdio.h>
#include <cuda_runtime.h>
#include <nvfunctional>
#include "Message.h"
#include "user.h"

#define EM_CNT_MAX 1024 // Maximum number of entry methods
#define SM_CNT 80

__device__ int entry_methods[EM_CNT_MAX];
__device__ Message* message_queue[SM_CNT]; // FIXME: Hard-coded # of SMs
__device__ int terminate[SM_CNT];

__device__ uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

__device__ bool check_terminate() {
  for (int i = 0; i < SM_CNT; i++) {
    if (terminate[i] == 0) {
      // TODO: Why infinite loop without this print?
      printf("SM %d not terminated\n", i);
      return false;
    }
  }
  return true;
}

__device__ void send(int sm, Message* msg) {
  message_queue[sm] = msg;
}

__device__ void recv(int my_sm) {
  Message* msg = message_queue[my_sm];
  if (msg) {
    // TODO: Handle received message
    printf("SM %d received message from SM %d\n", my_sm, msg->src_sm);
    msg = nullptr;

    // TODO: Terminate only when a termination message is received
    terminate[my_sm] = 1;
  }
}

__global__ void scheduler(int* sm_ids) {
  const int my_sm = get_smid();
  register_entry_methods(entry_methods);

  if (threadIdx.x == 0) {
    // Store SM ID
    sm_ids[blockIdx.x] = my_sm;

    int peer_sm = (my_sm+1) % SM_CNT;

    // Send a message to the peer SM
    Message* msg;
    msg = (Message*)malloc(sizeof(Message));
    msg->ep = 1;
    msg->src_sm = my_sm;
    msg->data = nullptr;
    send(peer_sm, msg);

    // Faux scheduler loop
    do {
      recv(my_sm);
    } while (message_queue != nullptr && !check_terminate());
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
