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

#define CHARE_TYPE_CNT_MAX 1024 // Maximum number of chares
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

__device__ ringbuf_t* rbuf;
__device__ size_t rbuf_size;
__device__ single_ringbuf_t* mbuf;
__device__ size_t mbuf_size;

__device__ ChareType* chare_types[CHARE_TYPE_CNT_MAX];
//__device__ EntryMethod* entry_methods[EM_CNT_MAX];

__device__ inline Envelope* createEnvelope(MsgType type, size_t msg_size,
                                           single_ringbuf_t* mbuf, size_t mbuf_size) {
  // Secure region in my message pool
  ringbuf_off_t mret = single_ringbuf_acquire(mbuf, msg_size);
  assert(mret != -1 && mret < mbuf_size);

  // Create envelope
  Envelope* env = new (mbuf->addr(mret)) Envelope(type, msg_size, nvshmem_my_pe());

  return env;
}

__device__ inline void sendMsg(Envelope* env, size_t msg_size, int dst_pe) {
  single_ringbuf_produce(mbuf);

  // Secure region in destination PE's message queue
  ringbuf_off_t rret;
  while ((rret = ringbuf_acquire(rbuf, msg_size, dst_pe)) == -1) {}
  assert(rret < rbuf_size);
  printf("PE %d: acquired %llu, msg size %llu\n", nvshmem_my_pe(), rret, msg_size);

  // Send message
  nvshmem_char_put((char*)rbuf->addr(rret), (char*)env, env->size, dst_pe);
  nvshmem_quiet();
  ringbuf_produce(rbuf, dst_pe);

  // Free region in my message pool
  size_t len, off;
  len = single_ringbuf_consume(mbuf, &off);
  single_ringbuf_release(mbuf, len);
}

__device__ inline void sendTermMsg(int dst_pe) {
  size_t msg_size = Envelope::alloc_size(0);
  Envelope* env = createEnvelope(MsgType::Terminate, msg_size, mbuf, mbuf_size);

  sendMsg(env, msg_size, dst_pe);
}

__device__ inline void sendRegMsg(int chare_id, int ep_id,
                                  size_t payload_size, int dst_pe) {
  size_t msg_size = Envelope::alloc_size(sizeof(RegularMsg) + payload_size);
  Envelope* env = createEnvelope(MsgType::Regular, msg_size, mbuf, mbuf_size);

  RegularMsg* reg_msg = new ((char*)env + sizeof(Envelope)) RegularMsg(chare_id, ep_id);
  // TODO: Fill in payload

  sendMsg(env, msg_size, dst_pe);
}

__device__ inline void sendCreateMsg(int dst_pe) {
}

__device__ inline ssize_t next_msg(void* addr, bool term_flags[]) {
  Envelope* env = (Envelope*)addr;
  printf("PE %d received msg type %d size %llu PE %d\n", nvshmem_my_pe(), env->type, env->src_pe);
  if (env->type == MsgType::Regular) {
    RegularMsg* reg_msg = (RegularMsg*)((char*)env + sizeof(Envelope));
    printf("PE %d regular message chare ID %d EP ID %d\n", nvshmem_my_pe(), reg_msg->chare_id, reg_msg->ep_id);
  } else if (env->type == MsgType::Terminate) {
    printf("PE %d terminate from PE %d\n", nvshmem_my_pe(), env->src_pe);
    term_flags[env->src_pe] = true;
  }

  return env->size;
}

__device__ inline void recv(bool term_flags[]) {
  size_t len, off;
  if ((len = ringbuf_consume(rbuf, &off)) != 0) {
    // Retrieved a contiguous range, there could be multiple messages
    size_t rem = len;
    ssize_t ret;
    while (rem) {
      printf("PE %d, next msg at offset %llu\n", nvshmem_my_pe(), off);
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

__global__ void scheduler() {
  if (!blockIdx.x && !threadIdx.x) {
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Register all chares and entry methods
    register_chare_types(chare_types);

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
      sendRegMsg(my_pe, 0, 128, dst_pe);
      sendTermMsg(dst_pe);
    } else {
      // Receive messages and terminate
      bool* term_flags = (bool*)malloc(sizeof(bool) * n_pes);
      for (int i = 0; i < n_pes; i++) {
        term_flags[i] = false;
      }
      term_flags[my_pe] = true;
      do {
        recv(term_flags);
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
  size_t h_rbuf_size = (1 << 10);
  ringbuf_t* h_rbuf = ringbuf_malloc(h_rbuf_size);
  size_t h_mbuf_size = (1 << 28);
  single_ringbuf_t* h_mbuf = single_ringbuf_malloc(h_mbuf_size);
  nvshmem_barrier_all();

  // Launch scheduler
  int grid_size = (argc > 1) ? atoi(argv[1]) : 1;
  int block_size = (argc > 2) ? atoi(argv[2]) : 1;
  if (!rank) {
    printf("NVCHARM\nGrid size: %d\nBlock size: %d\n", grid_size, block_size);
  }
  //void* scheduler_args[4] = { &rbuf, &rbuf_size, &mbuf, &mbuf_size };
  cudaMemcpyToSymbolAsync(rbuf, &h_rbuf, sizeof(ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(rbuf_size, &h_rbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf, sizeof(single_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  nvshmemx_collective_launch((const void*)scheduler, grid_size, block_size,
      //scheduler_args, 0, stream);
      nullptr, 0, stream);
  cuda_check_error();
  cudaStreamSynchronize(stream);
  //nvshmemx_barrier_all_on_stream(stream); // Hangs
  nvshmem_barrier_all();

  // Finalize NVSHMEM and MPI
  single_ringbuf_free(h_mbuf);
  ringbuf_free(h_rbuf);
  nvshmem_finalize();
  cudaStreamDestroy(stream);
  MPI_Finalize();

  return 0;
}

__device__ ChareType::ChareType(int id_) : id(id_) {}

template <typename T>
__device__ Chare<T>::Chare(int id_) : ChareType(id_), obj(nullptr) {}

template <typename T>
__device__ void Chare<T>::create(const T& obj_, int pe) {
  /*
  // Create one object for myself (PE 0)
  obj = new T(obj_);

  // TODO: Send creation messages to all PEs
  size_t payload_size = obj_.pack_size();
  size_t msg_size = Envelope::alloc_size(sizeof(CreateMsg) + payload_size);
  Envelope* env = createEnvelope(MsgType::Create, msg_size, mbuf, mbuf_size);
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
