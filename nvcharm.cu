#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include "Message.h"
#include "user.h"
#include "nvcharm.h"

#define DEBUG 0

#define MSG_CNT_MAX 1e6 // Maximum number of messages per PE
#define EM_CNT_MAX 1024 // Maximum number of entry methods

__device__ Message* msg_queue_symbol;
__device__ int* msg_queue_head_symbol;
__device__ int* msg_queue_tail_symbol;
__device__ EntryMethod* entry_methods[EM_CNT_MAX];

__device__ inline void send(int dst_pe, Message* msg) {
  /*
  int msg_idx = atomicAdd(&msg_cnt[sm], 1);
  msg_queue[MSG_IDX(sm,msg_idx)] = msg;
  */
}

__device__ inline void recv(int my_pe, bool& terminate) {
  /*
  int* head_ptr = nvshmem_ptr(msg_queue_tail_symbol, my_pe);
  if (msg) {

    if (msg->ep == -1) {
      terminate = true;
    }

    // Handle received message
    entry_methods[msg->ep]->call();

    msg = nullptr;
    processed++;
  }
  */
}

/*
// FIXME: Hard-coded limits
#define EM_CNT_MAX 1024 // Maximum number of entry methods
#define SM_CNT 80 // Number of SMs
#define MSG_CNT_MAX 1024 // Maximum number of messages in message queue
#define MSG_IDX(sm,idx) (MSG_CNT_MAX*(sm) + (idx))
#define CHARE_CNT_MAX 1024 // Maxinum number of chare types

__device__ ChareType* chare_types[SM_CNT * CHARE_CNT_MAX];
__device__ int chare_cnt[SM_CNT];
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
*/

/*
__global__ void simple_shift(int *destination) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;

  if (!blockIdx.x && !threadIdx.x) {
    nvshmem_int_p(destination, mype, peer);
    nvshmem_barrier_all();
  }
}
*/

__global__ void scheduler(Message* msg_queue, int* msg_queue_head, int* msg_queue_tail) {
  if (!blockIdx.x && !threadIdx.x) {
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    bool terminate = false;

    // Register user's entry methods
    register_entry_methods(entry_methods);

    // Execute user's main function
    charm_main();

    // Scheduler loop
    do {
      recv(my_pe, terminate);
    } while (!terminate);
  }
}

int main(int argc, char* argv[]) {
  int rank, msg;
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

  // NVSHMEM symmetric memory allocations
  Message* msg_queue = (Message*) nvshmem_malloc(sizeof(Message) * MSG_CNT_MAX);
  int* msg_queue_head = (int*) nvshmem_malloc(sizeof(int));
  int* msg_queue_tail = (int*) nvshmem_malloc(sizeof(int));
  nvshmem_barrier_all();
  cudaMemcpyToSymbol(msg_queue_symbol, msg_queue, sizeof(Message*));
  cudaMemcpyToSymbol(msg_queue_head_symbol, msg_queue_head, sizeof(int*));
  cudaMemcpyToSymbol(msg_queue_tail_symbol, msg_queue_tail, sizeof(int*));

  // Launch scheduler
  int grid_size = (argc > 1) ? atoi(argv[1]) : 1;
  int block_size = (argc > 2) ? atoi(argv[2]) : 1;
  if (!rank) {
    printf("NVCHARM\nGrid size: %d\nBlock size: %d\n", grid_size, block_size);
  }
  void* scheduler_args[3] = { &msg_queue, &msg_queue_head, &msg_queue_tail };
  nvshmemx_collective_launch((const void*)scheduler, grid_size, block_size,
      scheduler_args, 0, stream);
  cuda_check_error();
  cudaStreamSynchronize(stream);
  //nvshmemx_barrier_all_on_stream(stream); // Hangs
  nvshmem_barrier_all();

  // Finalize NVSHMEM and MPI
  nvshmem_free(msg_queue);
  nvshmem_finalize();
  cudaStreamDestroy(stream);
  MPI_Finalize();

  return 0;
}

template <typename T>
__device__ Chare<T>::Chare(T obj_, int n_chares_) : obj(obj_), n_chares(n_chares_) {
  /*
  // TODO: Create chare objects on all GPUs
  mapping = new Mapping[SM_CNT];
  int rem = n_chares % SM_CNT;
  int start_idx = 0;
  for (int i = 0; i < SM_CNT; i++) {
    int n_chares_sm = n_chares / SM_CNT;
    if (i < rem) n_chares_sm++;
    mapping[i].sm_id = i;
    mapping[i].start_idx = start_idx;
    mapping[i].end_idx = start_idx + n_chares_sm - 1;
    start_idx += n_chares_sm;

    //CreationMessage<T>* create_msg = new CreationMessage<T>(obj);
  }
  */
}

template <typename T>
__device__ void Chare<T>::invoke(int ep, int idx) {
  /*
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
  */
}
