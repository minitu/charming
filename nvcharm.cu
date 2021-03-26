#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include "nvcharm.h"
#include "message.h"
#include "ringbuf.h"
#include "util.h"
#include "user.h"

#define CHARE_TYPE_CNT_MAX 1024 // Maximum number of chare types

__device__ mpsc_ringbuf_t* rbuf;
__device__ size_t rbuf_size;
__device__ spsc_ringbuf_t* mbuf;
__device__ size_t mbuf_size;

__device__ ChareType* chare_types[CHARE_TYPE_CNT_MAX];

__device__ inline Envelope* create_envelope(MsgType type, size_t msg_size) {
  // Secure region in my message pool
  ringbuf_off_t mret = spsc_ringbuf_acquire(mbuf, msg_size);
  assert(mret != -1 && mret < mbuf_size);

  // Create envelope
  Envelope* env = new (mbuf->addr(mret)) Envelope(type, msg_size, nvshmem_my_pe());

  return env;
}

__device__ inline void send_msg(Envelope* env, size_t msg_size, int dst_pe) {
  spsc_ringbuf_produce(mbuf);

  // Secure region in destination PE's message queue
  ringbuf_off_t rret;
  while ((rret = mpsc_ringbuf_acquire(rbuf, msg_size, dst_pe)) == -1) {}
  assert(rret < rbuf_size);

  // Send message
  nvshmem_char_put((char*)rbuf->addr(rret), (char*)env, env->size, dst_pe);
  nvshmem_quiet();
  mpsc_ringbuf_produce(rbuf, dst_pe);

  // Free region in my message pool
  size_t len, off;
  len = spsc_ringbuf_consume(mbuf, &off);
  spsc_ringbuf_release(mbuf, len);
}

__device__ inline void send_term_msg(int dst_pe) {
  size_t msg_size = Envelope::alloc_size(0);
  Envelope* env = create_envelope(MsgType::Terminate, msg_size);

  send_msg(env, msg_size, dst_pe);
}

__device__ inline void send_reg_msg(int chare_id, int ep_id,
                                  size_t payload_size, int dst_pe) {
  size_t msg_size = Envelope::alloc_size(sizeof(RegularMsg) + payload_size);
  Envelope* env = create_envelope(MsgType::Regular, msg_size);

  RegularMsg* reg_msg = new ((char*)env + sizeof(Envelope)) RegularMsg(chare_id, ep_id);
  // TODO: Fill in payload

  send_msg(env, msg_size, dst_pe);
}

__device__ inline ssize_t next_msg(void* addr, bool& term_flag) {
  Envelope* env = (Envelope*)addr;
#ifdef DEBUG
  printf("PE %d received msg type %d size %llu from PE %d\n", nvshmem_my_pe(), env->type, env->size, env->src_pe);
#endif

  if (env->type == MsgType::Create) {
    // Creation message
    CreateMsg* create_msg = (CreateMsg*)((char*)env + sizeof(Envelope));
#ifdef DEBUG
    printf("PE %d creation msg chare ID %d\n", nvshmem_my_pe(), create_msg->chare_id);
#endif
    ChareType*& chare_type = chare_types[create_msg->chare_id];
    chare_type->alloc();
    chare_type->unpack((char*)create_msg + sizeof(CreateMsg));
  } else if (env->type == MsgType::Regular) {
    // Regular message
    RegularMsg* reg_msg = (RegularMsg*)((char*)env + sizeof(Envelope));
#ifdef DEBUG
    printf("PE %d regular msg chare ID %d EP ID %d\n", nvshmem_my_pe(), reg_msg->chare_id, reg_msg->ep_id);
#endif
    // TODO: Chare ID needs to be fixed
    ChareType*& chare_type = chare_types[0];
    chare_type->call(reg_msg->ep_id);
  } else if (env->type == MsgType::Terminate) {
    // Termination message
#ifdef DEBUG
    printf("PE %d terminate msg\n", nvshmem_my_pe(), env->src_pe);
#endif
    term_flag = true;
  }

  return env->size;
}

__device__ inline void recv(bool &term_flag) {
  size_t len, off;
  if ((len = mpsc_ringbuf_consume(rbuf, &off)) != 0) {
    // Retrieved a contiguous range, there could be multiple messages
    size_t rem = len;
    ssize_t ret;
    while (rem) {
      ret = next_msg(rbuf->addr(off), term_flag);
      off += ret;
      rem -= ret;
    }
    mpsc_ringbuf_release(rbuf, len);
  }
}

__global__ void scheduler() {
  if (!blockIdx.x && !threadIdx.x) {
    bool term_flag = false;
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    // Register all chares and entry methods
    register_chare_types(chare_types);

    // Initialize message queue
    mpsc_ringbuf_init(rbuf, rbuf_size);
    spsc_ringbuf_init(mbuf, mbuf_size);

    nvshmem_barrier_all();

    if (my_pe == 0) {
      // Execute user's main function
      charm_main(chare_types);
    }

    nvshmem_barrier_all();

    // Receive messages and terminate
    do {
      recv(term_flag);
    } while(!term_flag);
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
  mpsc_ringbuf_t* h_rbuf = mpsc_ringbuf_malloc(h_rbuf_size);
  size_t h_mbuf_size = (1 << 28);
  spsc_ringbuf_t* h_mbuf = spsc_ringbuf_malloc(h_mbuf_size);
  nvshmem_barrier_all();

  // Launch scheduler
  int grid_size = (argc > 1) ? atoi(argv[1]) : 1;
  int block_size = (argc > 2) ? atoi(argv[2]) : 1;
  if (!rank) {
    printf("NVCHARM\nGrid size: %d\nBlock size: %d\n", grid_size, block_size);
  }
  //void* scheduler_args[4] = { &rbuf, &rbuf_size, &mbuf, &mbuf_size };
  cudaMemcpyToSymbolAsync(rbuf, &h_rbuf, sizeof(mpsc_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(rbuf_size, &h_rbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf, sizeof(spsc_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  nvshmemx_collective_launch((const void*)scheduler, grid_size, block_size,
      //scheduler_args, 0, stream);
      nullptr, 0, stream);
  cuda_check_error();
  cudaStreamSynchronize(stream);
  //nvshmemx_barrier_all_on_stream(stream); // Hangs
  nvshmem_barrier_all();

  // Finalize NVSHMEM and MPI
  spsc_ringbuf_free(h_mbuf);
  mpsc_ringbuf_free(h_rbuf);
  nvshmem_finalize();
  cudaStreamDestroy(stream);
  MPI_Finalize();

  return 0;
}

// TODO: Currently 1 chare per PE
template <typename T>
__device__ void Chare<T>::create(T& obj_) {
  // Create one object for myself (PE 0)
  alloc(obj_);

  // Send creation messages to all other PEs
  int my_pe = nvshmem_my_pe();
  for (int pe = 0; pe < nvshmem_n_pes(); pe++) {
    if (pe == my_pe) continue;

    size_t payload_size = obj_.pack_size();
    size_t msg_size = Envelope::alloc_size(sizeof(CreateMsg) + payload_size);
    Envelope* env = create_envelope(MsgType::Create, msg_size);

    CreateMsg* create_msg = new ((char*)env + sizeof(Envelope)) CreateMsg(id);
    obj_.pack((char*)create_msg + sizeof(CreateMsg));

    send_msg(env, msg_size, pe);
  }
}

// TODO
// - Change to send to chare instead of PE
// - Support entry method parameters (single buffer for now)
// Note: Chare should have been already created at this PE via a creation message
template <typename T>
__device__ void Chare<T>::invoke(int idx, int ep) {
  if (idx == -1) {
    // TODO: Broadcast to all chares
  } else {
    // Send a regular message to the target PE
    send_reg_msg(idx, ep, 0, idx);
  }
}

__device__ void ckExit() {
  int n_pes = nvshmem_n_pes();
  for (int pe = 0; pe < n_pes; pe++) {
    send_term_msg(pe);
  }
}
