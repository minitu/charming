#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include "message.h"
#include "user.h"
#include "nvcharm.h"
#include "ringbuf.h"

#define DEBUG 0

#define EM_CNT_MAX 1024 // Maximum number of entry methods

/*
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
*/

__device__ EntryMethod* entry_methods[EM_CNT_MAX];

__device__ inline void send(int chare_id, int ep_id, int dst_pe,
                            ringbuf_t* rbuf, size_t rbuf_size,
                            single_ringbuf_t* mbuf, size_t mbuf_size) {
  // Secure region in destination PE's message queue
  ringbuf_off_t rret, mret;
  size_t msg_size = Message::alloc_size(0);
  while ((rret = ringbuf_acquire(rbuf, msg_size, dst_pe)) == -1) {}
  assert(rret < rbuf_size);
  printf("PE %d: acquired %llu, msg size %llu\n", nvshmem_my_pe(), rret, msg_size);

  // Secure region in my message pool
  mret = single_ringbuf_acquire(mbuf, msg_size);
  assert(mret != -1 && mret < mbuf_size);

  // Populate message
  Message* msg = new (mbuf->addr(mret)) Message(0, chare_id, ep_id);
  single_ringbuf_produce(mbuf);

  // Send message
  nvshmem_char_put((char*)rbuf->addr(rret), (char*)msg, msg->size, dst_pe);
  nvshmem_quiet();
  ringbuf_produce(rbuf, dst_pe);

  // Free region in my message pool
  size_t len, off;
  len = single_ringbuf_consume(mbuf, &off);
  single_ringbuf_release(mbuf, len);
}

__device__ inline ssize_t next_msg(void* addr, bool term_flags[]) {
  Message* msg = (Message*)addr;
  printf("PE %d received msg size %llu chare_id %d ep_id %d\n",
      nvshmem_my_pe(), msg->size, msg->chare_id, msg->ep_id);
  if (msg->ep_id == -1) term_flags[msg->chare_id] = true;

  return msg->size;
}

__device__ inline void recv(ringbuf_t* rbuf, bool term_flags[]) {
  size_t len, off;
  if ((len = ringbuf_consume(rbuf, &off)) != 0) {
    // Retrieved a contiguous range, there could be multiple messages
    size_t rem = len;
    ssize_t ret;
    while (rem) {
      ret = next_msg(rbuf->addr(off), term_flags);
      off += ret;
      rem -= ret;
    }
    ringbuf_release(rbuf, len);
  }
}

__device__ bool check_terminate(bool flags[], int cnt) {
  for (int i = 0; i < cnt; i++) {
    if (!flags[i]) return false;
  }
  return true;
}

__global__ void scheduler(ringbuf_t* rbuf, size_t rbuf_size,
                          single_ringbuf_t* mbuf, size_t mbuf_size) {
  if (!blockIdx.x && !threadIdx.x) {
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Register user's entry methods
    register_entry_methods(entry_methods);

    // Initialize message queue
    ringbuf_init(rbuf, rbuf_size);
    single_ringbuf_init(mbuf, mbuf_size);

    nvshmem_barrier_all();

    if (my_pe == 0) {
      // Execute user's main function
      charm_main();
    }

    nvshmem_barrier_all();

    // XXX: Testing
    if (my_pe != 0) {
      int dst_pe = 0;
      send(my_pe, 0, dst_pe, rbuf, rbuf_size, mbuf, mbuf_size);
      send(my_pe, -1, dst_pe, rbuf, rbuf_size, mbuf, mbuf_size);
    } else {
      // Receive messages and terminate
      bool* term_flags = (bool*)malloc(sizeof(bool) * n_pes);
      for (int i = 0; i < n_pes; i++) {
        term_flags[i] = false;
      }
      term_flags[my_pe] = true;
      do {
        recv(rbuf, term_flags);
      } while(!check_terminate(term_flags, n_pes));
    }
  }
}

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

  // Allocate message queue with NVSHMEM
  size_t rbuf_size = (1 << 29);
  ringbuf_t* rbuf = ringbuf_malloc(rbuf_size);
  size_t mbuf_size = (1 << 28);
  single_ringbuf_t* mbuf = single_ringbuf_malloc(mbuf_size);
  nvshmem_barrier_all();

  // Launch scheduler
  int grid_size = (argc > 1) ? atoi(argv[1]) : 1;
  int block_size = (argc > 2) ? atoi(argv[2]) : 1;
  if (!rank) {
    printf("NVCHARM\nGrid size: %d\nBlock size: %d\n", grid_size, block_size);
  }
  void* scheduler_args[4] = { &rbuf, &rbuf_size, &mbuf, &mbuf_size };
  nvshmemx_collective_launch((const void*)scheduler, grid_size, block_size,
      scheduler_args, 0, stream);
  cuda_check_error();
  cudaStreamSynchronize(stream);
  //nvshmemx_barrier_all_on_stream(stream); // Hangs
  nvshmem_barrier_all();

  // Finalize NVSHMEM and MPI
  single_ringbuf_free(mbuf);
  ringbuf_free(rbuf);
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
