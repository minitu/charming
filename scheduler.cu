#include "scheduler.h"
#include "nvcharm.h"
#include "chare.h"
#include "ringbuf.h"

extern __device__ mpsc_ringbuf_t* rbuf;
extern __device__ size_t rbuf_size;
extern __device__ spsc_ringbuf_t* mbuf;
extern __device__ size_t mbuf_size;

extern __device__ ChareType* chare_types[];

__device__ Envelope* create_envelope(MsgType type, size_t msg_size) {
  // Secure region in my message pool
  ringbuf_off_t mret = spsc_ringbuf_acquire(mbuf, msg_size);
  assert(mret != -1 && mret < mbuf_size);

  // Create envelope
  Envelope* env = new (mbuf->addr(mret)) Envelope(type, msg_size, nvshmem_my_pe());

  return env;
}

__device__ void send_msg(Envelope* env, size_t msg_size, int dst_pe) {
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

__device__ void send_term_msg(int dst_pe) {
  size_t msg_size = Envelope::alloc_size(0);
  Envelope* env = create_envelope(MsgType::Terminate, msg_size);

  send_msg(env, msg_size, dst_pe);
}

__device__ void send_reg_msg(int chare_id, int ep_id,
                             size_t payload_size, int dst_pe) {
  size_t msg_size = Envelope::alloc_size(sizeof(RegularMsg) + payload_size);
  Envelope* env = create_envelope(MsgType::Regular, msg_size);

  RegularMsg* reg_msg = new ((char*)env + sizeof(Envelope)) RegularMsg(chare_id, ep_id);
  // TODO: Fill in payload

  send_msg(env, msg_size, dst_pe);
}

__device__ __forceinline__ ssize_t next_msg(void* addr, bool& term_flag) {
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

__device__ __forceinline__ void recv(bool &term_flag) {
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

