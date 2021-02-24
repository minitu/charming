#include <stdio.h>
#include <cuda_runtime.h>
#include "Message.h"
#include "user.h"
#include "nvcharm.h"

#define DEBUG 1

// FIXME: Hard-coded limits
#define EM_CNT_MAX 1024 // Maximum number of entry methods
#define SM_CNT 80 // Number of SMs
#define MSG_CNT_MAX 1024 // Maximum number of messages in message queue
#define MSG_IDX(sm,idx) (MSG_CNT_MAX*(sm) + (idx))

__device__ EntryMethod* entry_methods[EM_CNT_MAX];
__device__ Message* msg_queue[SM_CNT * MSG_CNT_MAX];
__device__ int msg_cnt[SM_CNT];
__device__ int terminate[SM_CNT];

__device__ uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

using clock_value_t = long long;

__device__ void sleep(clock_value_t sleep_cycles) {
  clock_value_t start = clock64();
  clock_value_t cycles_elapsed;
  do {
    cycles_elapsed = clock64() - start;
  } while (cycles_elapsed < sleep_cycles);
}

__device__ void send(int sm, Message* msg) {
  int msg_idx = atomicAdd(&msg_cnt[sm], 1);
  msg_queue[MSG_IDX(sm,msg_idx)] = msg;
#if DEBUG
  printf("Stored message in idx %d, msg %p\n", MSG_IDX(sm,msg_idx), msg);
#endif
}

__device__ void recv(int my_sm, int& processed, bool& terminate) {
  // TODO: Recv doesn't happen without follownig print statement, why?
#if DEBUG
  printf("SM %d checking idx %d\n", my_sm, MSG_IDX(my_sm, processed));
#endif
  Message*& msg = msg_queue[MSG_IDX(my_sm, processed)];
  if (msg) {
#if DEBUG
    printf("SM %d received message %p, SM %d, ep %d\n",
        my_sm, msg, msg->src_sm, msg->ep);
#endif

    if (msg->ep == -1) {
      terminate = true;
#if DEBUG
      printf("SM %d terminating\n", my_sm);
#endif
    }

    // Handle received message
    entry_methods[msg->ep]->call();

    msg = nullptr;
    processed++;
  }
}

__global__ void scheduler(DeviceCtx* ctx) {
  //const int my_sm = get_smid();
  const int my_sm = blockIdx.x;
  __shared__ int processed;
  __shared__ bool terminate;

  register_entry_methods(entry_methods);

  // Leader thread in each thread block runs the scheduler loop
  if (threadIdx.x == 0) {
    processed = 0;
    terminate = false;

    if (blockIdx.x == 0) {
      printf("SMs: %d\n", ctx->n_sms);

      // Execute user's main function
      charm_main();
    }

    // Scheduler loop
    do {
      recv(my_sm, processed, terminate);
      //sleep(1);
    } while (!terminate);
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

  // Create device context
  DeviceCtx* ctx;
  cudaMallocManaged(&ctx, sizeof(DeviceCtx));
  ctx->n_sms = prop.multiProcessorCount;

  // Obtain kernel block and grid sizes
  int block_size = 1;
  //int grid_size = prop.multiProcessorCount;
  int grid_size = 10;
  if (argc > 1) block_size = atoi(argv[1]);
  if (argc > 2) grid_size = atoi(argv[2]);
  printf("* Test properties\n"
      "Block size: %d\nGrid size: %d\n\n", block_size, grid_size);

  // Run kernel
  scheduler<<<grid_size, block_size>>>(ctx);
  cudaDeviceSynchronize();

  return 0;
}

/******************** Chare ********************/
__device__ ChareArray::ChareArray(int n_chares_) {
  n_chares = n_chares_;
  mapping = (int*)malloc(sizeof(int) * n_chares);
  for (int i = 0; i < n_chares; i++) {
    mapping[i] = i % SM_CNT; // FIXME
  }
}

__device__ void ChareArray::invoke(int ep, int idx) {
  if (idx == -1) {
    // Broadcast to all chares
    for (int i = 0; i < n_chares; i++) {
      Message* msg = (Message*)malloc(sizeof(Message));
      msg->ep = ep;
      int target_sm = mapping[i];
      send(target_sm, msg);
    }
  } else {
    // P2P
    Message* msg = (Message*)malloc(sizeof(Message));
    msg->ep = ep;
    int target_sm = mapping[idx];
    send(target_sm, msg);
  }
}
