#include "scheduler.h"
#include "nvcharm.h"
#include "chare.h"
#include "ringbuf.h"

using namespace charm;

extern __device__ mpsc_ringbuf_t* rbuf;
extern __device__ size_t rbuf_size;
extern __device__ spsc_ringbuf_t* mbuf;
extern __device__ size_t mbuf_size;

extern __device__ chare_type* chare_types[];

__device__ envelope* charm::create_envelope(msgtype type, size_t msg_size) {
  // Secure region in my message pool
  ringbuf_off_t mret = spsc_ringbuf_acquire(mbuf, msg_size);
  assert(mret != -1 && mret < mbuf_size);

  // Create envelope
  envelope* env = new (mbuf->addr(mret)) envelope(type, msg_size, nvshmem_my_pe());

  return env;
}

__device__ void charm::send_msg(envelope* env, size_t msg_size, int dst_pe) {
  spsc_ringbuf_produce(mbuf);

  // Secure region in destination PE's message queue
  ringbuf_off_t rret;
  while ((rret = mpsc_ringbuf_acquire(rbuf, msg_size, dst_pe)) == -1) {}
  assert(rret < rbuf_size);

  // Send message
  printf("acquired %lld, sending to PE %d rbuf->addr(rret): %p, msg size %lld\n", rret, dst_pe, rbuf->addr(rret), env->size);
  nvshmem_char_put((char*)rbuf->addr(rret), (char*)env, env->size, dst_pe);
  nvshmem_quiet();
  mpsc_ringbuf_produce(rbuf, dst_pe);

  // Free region in my message pool
  size_t len, off;
  len = spsc_ringbuf_consume(mbuf, &off);
  spsc_ringbuf_release(mbuf, len);
}

__device__ void charm::send_reg_msg(int chare_id, int chare_idx, int ep_id,
                                    size_t payload_size, int dst_pe) {
  size_t msg_size = envelope::alloc_size(sizeof(charm::regular_msg) + payload_size);
  envelope* env = create_envelope(msgtype::regular, msg_size);

  charm::regular_msg* msg = new ((char*)env + sizeof(envelope)) charm::regular_msg(chare_id, chare_idx, ep_id);

  // TODO: Fill in payload

  send_msg(env, msg_size, dst_pe);
}

__device__ void charm::send_term_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  envelope* env = create_envelope(msgtype::terminate, msg_size);

  send_msg(env, msg_size, dst_pe);
}

__device__ __forceinline__ ssize_t next_msg(void* addr, bool& term_flag) {
  envelope* env = (envelope*)addr;
#ifdef DEBUG
  printf("PE %d received msg type %d size %llu from PE %d\n",
         nvshmem_my_pe(), env->type, env->size, env->src_pe);
#endif

  if (env->type == msgtype::create) {
    // Creation message
    charm::create_msg* msg = (charm::create_msg*)((char*)env + sizeof(envelope));
#ifdef DEBUG
    printf("PE %d creation msg chare ID %d, start idx %d, end idx %d\n",
           nvshmem_my_pe(), msg->chare_id, msg->start_idx, msg->end_idx);
#endif
    charm::chare_type*& chare_type = chare_types[msg->chare_id];
    chare_type->alloc(msg->n_chares);
    void* packed_data = (char*)msg + sizeof(charm::create_msg);
    for (int i = 0; i < msg->n_chares; i++) {
      chare_type->unpack(packed_data, i);
    }
  } else if (env->type == msgtype::regular) {
    // Regular message
    charm::regular_msg* msg = (charm::regular_msg*)((char*)env + sizeof(envelope));
#ifdef DEBUG
    printf("PE %d regular msg chare ID %d chare idx %d EP ID %d\n", nvshmem_my_pe(), msg->chare_id, msg->chare_idx, msg->ep_id);
#endif
    charm::chare_type*& chare_type = chare_types[msg->chare_id];
    chare_type->call(msg->chare_idx, msg->ep_id);
  } else if (env->type == msgtype::terminate) {
    // Termination message
#ifdef DEBUG
    printf("PE %d terminate msg\n", nvshmem_my_pe());
#endif
    term_flag = true;
  }

  return env->size;
}

__device__ __forceinline__ void recv_msg(bool &term_flag) {
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

__global__ void charm::scheduler() {
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
      main(chare_types);
    }

    nvshmem_barrier_all();

    // Receive messages and terminate
    do {
      recv_msg(term_flag);
    } while(!term_flag);
  }
}
