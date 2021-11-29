#include <nvshmem.h>
#include <nvshmemx.h>

#include "scheduler.h"
#include "charming.h"
#include "chare.h"
#include "ringbuf.h"
#include "util.h"

using namespace charm;

extern __constant__ int c_my_pe;
extern __constant__ int c_n_pes;

extern __device__ spsc_ringbuf_t* mbuf;
extern __device__ size_t mbuf_size;
extern __device__ uint64_t* used_arr;
extern __device__ uint64_t* addr_arr;
extern __device__ uint64_t* size_arr;
extern __device__ size_t arr_size;

extern __device__ chare_proxy_base* chare_proxies[];

__device__ envelope* charm::create_envelope(msgtype type, size_t msg_size) {
  // Reserve space for this message in message buffer
  ringbuf_off_t mret = spsc_ringbuf_acquire(mbuf, msg_size);
  assert(mret != -1 && mret < mbuf_size);

  // Create envelope
  return new (mbuf->addr(mret)) envelope(type, msg_size, c_my_pe);
}

__device__ void charm::send_msg(envelope* env, size_t msg_size, int dst_pe) {
  // Message is ready to be sent
  spsc_ringbuf_produce(mbuf);

  // Obtain a message index for the target PE and set it to 1
  size_t offset = MSG_IN_FLIGHT_MAX * dst_pe;
  uint64_t* my_used_arr = used_arr + offset;
  size_t msg_idx = nvshmem_uint64_wait_until_any(my_used_arr, MSG_IN_FLIGHT_MAX,
      nullptr, NVSHMEM_CMP_EQ, 0);
  assert(msg_idx != SIZE_MAX);
  nvshmemx_signal_op(my_used_arr + msg_idx, 1, NVSHMEM_SIGNAL_SET, c_my_pe);
#ifdef DEBUG
  printf("PE %d sending message request to PE %d, message index %llu, addr %p, size %llu\n",
      c_my_pe, dst_pe, msg_idx, env, msg_size);
#endif

  // Send address of source buffer and message size
  // TODO: Send as one 128-bit buffer?
  offset = MSG_IN_FLIGHT_MAX * c_my_pe;
  uint64_t* my_addr_arr = addr_arr + offset;
  uint64_t* my_size_arr = size_arr + offset;
  nvshmemx_signal_op(my_addr_arr + msg_idx, (uint64_t)env, NVSHMEM_SIGNAL_SET, dst_pe);
  nvshmemx_signal_op(my_size_arr + msg_idx, (uint64_t)msg_size, NVSHMEM_SIGNAL_SET, dst_pe);

  // TODO: Needs to be done in comm TB loop
  /*
  // Free region in my message pool
  size_t len, off;
  len = spsc_ringbuf_consume(mbuf, &off);
  spsc_ringbuf_release(mbuf, len);
  */
}

__device__ void charm::send_dummy_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  envelope* env = create_envelope(msgtype::dummy, msg_size);

  send_msg(env, msg_size, dst_pe);
}

__device__ void charm::send_reg_msg(int chare_id, int chare_idx, int ep_id,
                                    void* buf, size_t payload_size, int dst_pe) {
  size_t msg_size = envelope::alloc_size(sizeof(regular_msg) + payload_size);
  envelope* env = create_envelope(msgtype::regular, msg_size);

  regular_msg* msg = new ((char*)env + sizeof(envelope)) regular_msg(chare_id, chare_idx, ep_id);

  // Fill in payload (from regular GPU memory to NVSHMEM symmetric memory)
  if (payload_size > 0) {
    memcpy((char*)msg + sizeof(regular_msg), buf, payload_size);
  }

  send_msg(env, msg_size, dst_pe);
}

__device__ void charm::send_begin_term_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  envelope* env = create_envelope(msgtype::begin_terminate, msg_size);

  send_msg(env, msg_size, dst_pe);
}

__device__ void charm::send_do_term_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  envelope* env = create_envelope(msgtype::do_terminate, msg_size);

  send_msg(env, msg_size, dst_pe);
}

__device__ __forceinline__ ssize_t next_msg(void* addr, bool& begin_term_flag,
                                            bool& do_term_flag) {
  static int dummy_cnt = 0;
  static clock_value_t start;
  static clock_value_t end;
  envelope* env = (envelope*)addr;
#ifdef DEBUG
  printf("PE %d received msg type %d size %llu from PE %d\n",
         nvshmem_my_pe(), env->type, env->size, env->src_pe);
#endif

  if (env->type == msgtype::dummy) {
    // Dummy message
    if (dummy_cnt == 0) {
      start = clock64();
    } else if (dummy_cnt == DUMMY_ITERS-1) {
      end = clock64();
      printf("Receive avg clocks: %lld\n", (end - start) / DUMMY_ITERS);
    }
    dummy_cnt++;
  } else if (env->type == msgtype::create) {
    // Creation message
    create_msg* msg = (create_msg*)((char*)env + sizeof(envelope));
#ifdef DEBUG
    printf("PE %d creation msg chare ID %d, n_local %d, n_total %d, start idx %d, end idx %d\n",
           nvshmem_my_pe(), msg->chare_id, msg->n_local, msg->n_total, msg->start_idx, msg->end_idx);
#endif
    chare_proxy_base*& chare_proxy = chare_proxies[msg->chare_id];
    chare_proxy->alloc(msg->n_local, msg->n_total, msg->start_idx, msg->end_idx);
    char* tmp = (char*)msg + sizeof(create_msg);
    chare_proxy->store_loc_map(tmp);
    tmp += sizeof(int) * msg->n_total;
    for (int i = 0; i < msg->n_local; i++) {
      chare_proxy->unpack(tmp, i);
    }
  } else if (env->type == msgtype::regular) {
    // Regular message
    regular_msg* msg = (regular_msg*)((char*)env + sizeof(envelope));
#ifdef DEBUG
    printf("PE %d regular msg chare ID %d chare idx %d EP ID %d\n", nvshmem_my_pe(), msg->chare_id, msg->chare_idx, msg->ep_id);
#endif
    chare_proxy_base*& chare_proxy = chare_proxies[msg->chare_id];
    void* payload = (char*)msg + sizeof(regular_msg);
    // TODO: Copy payload?
    chare_proxy->call(msg->chare_idx, msg->ep_id, payload);
  } else if (env->type == msgtype::begin_terminate) {
    // Should only be received by PE 0
    assert(my_pe() == 0);
    // Begin termination message
#ifdef DEBUG
    printf("PE %d begin terminate msg\n", nvshmem_my_pe());
#endif
    if (!begin_term_flag) {
      for (int i = 0; i < n_pes(); i++) {
        send_do_term_msg(i);
      }
      begin_term_flag = true;
    }
  } else if (env->type == msgtype::do_terminate) {
    // Do termination message
#ifdef DEBUG
    printf("PE %d do terminate msg\n", nvshmem_my_pe());
#endif
    do_term_flag = true;
  }

  return env->size;
}

__device__ __forceinline__ void recv_msg(bool& begin_term_flag, bool &do_term_flag) {
  // Check if there are any message requests
  size_t msg_idx = nvshmem_uint64_test_any(addr_arr, MSG_IN_FLIGHT_MAX * c_n_pes,
      nullptr, NVSHMEM_CMP_GT, 0);
  if (msg_idx != SIZE_MAX) {
    // Found a message request, obtain source PE, buffer address and message size
    uint64_t src_addr = nvshmem_signal_fetch(addr_arr + msg_idx);
    uint64_t src_size = nvshmem_signal_fetch(size_arr + msg_idx);
    int src_pe = msg_idx / c_n_pes;
    msg_idx -= MSG_IN_FLIGHT_MAX * src_pe;
#ifdef DEBUG
    printf("PE %d received message request from PE %d, message index %llu, addr %p, size %llu\n",
        c_my_pe, src_pe, msg_idx, (void*)src_addr, src_size);
#endif

    // Reserve space for incoming message
    ringbuf_off_t mret = spsc_ringbuf_acquire(mbuf, src_size);
    assert(mret != -1 && mret < mbuf_size);

    // Perform a remote get
    // TODO: Make asynchronous
    nvshmem_char_get((char*)mbuf->addr(mret), (char*)src_addr, src_size, src_pe);

    // Process message
    next_msg(mbuf->addr(mret), begin_term_flag, do_term_flag);

    // Clear message request
    nvshmemx_signal_op(addr_arr + MSG_IN_FLIGHT_MAX * src_pe + msg_idx, 0,
        NVSHMEM_SIGNAL_SET, c_my_pe);

    // Notify sender
    nvshmemx_signal_op(used_arr + MSG_IN_FLIGHT_MAX * c_my_pe + msg_idx, 0,
        NVSHMEM_SIGNAL_SET, src_pe);
  }

  // TODO: Garbage collect
}

__global__ void charm::scheduler(int argc, char** argv, size_t* argvs) {
  if (!blockIdx.x && !threadIdx.x) {
    bool begin_term_flag = false;
    bool do_term_flag = false;

    // Register user chares and entry methods on all PEs
    chare_proxy_cnt = 0;
    register_chares();

    // Initialize message queue
    spsc_ringbuf_init(mbuf, mbuf_size);

    nvshmem_barrier_all();

    if (c_my_pe == 0) {
      // Execute user's main function
      main(argc, argv, argvs);
    }

    nvshmem_barrier_all(); // FIXME: No need?

    // Receive messages and terminate
    do {
      recv_msg(begin_term_flag, do_term_flag);
    } while (!do_term_flag);

#if DEBUG
    printf("PE %d terminating...\n", c_my_pe);
#endif
  }
}
