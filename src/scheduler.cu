#include <nvshmem.h>
#include <nvshmemx.h>

#include "scheduler.h"
#include "charming.h"
#include "chare.h"
#include "ringbuf.h"
#include "heap.h"
#include "util.h"

using namespace charm;

extern __constant__ int c_my_pe;
extern __constant__ int c_n_pes;

extern __device__ spsc_ringbuf_t* mbuf;
extern __device__ size_t mbuf_size;
extern __device__ uint64_t* send_status;
extern __device__ uint64_t* recv_alloy;
extern __device__ uint64_t* send_alloy;
extern __device__ size_t* send_status_idx;
extern __device__ size_t* recv_alloy_idx;
extern __device__ alloy* heap_buf;
extern __device__ size_t heap_buf_size;

extern __device__ chare_proxy_base* chare_proxies[];

enum {
  SIGNAL_FREE = 0,
  SIGNAL_USED = 1,
  SIGNAL_CLUP = 2
};

__device__ envelope* charm::create_envelope(msgtype type, size_t msg_size, size_t* offset) {
  // Reserve space for this message in message buffer
  ringbuf_off_t mret = spsc_ringbuf_acquire(mbuf, msg_size);
  assert(mret != -1 && mret < mbuf_size);
  *offset = mret;

  // Create envelope
  return new (mbuf->addr(mret)) envelope(type, msg_size, c_my_pe);
}

__device__ void charm::send_msg(size_t offset, size_t msg_size, int dst_pe) {
  // Message is ready to be sent
  spsc_ringbuf_produce(mbuf);

  // Obtain a message index for the target PE and set the corresponding used signal
  size_t signal_offset = MSG_IN_FLIGHT_MAX * dst_pe;
  uint64_t* my_send_status = send_status + signal_offset;
  size_t msg_idx = nvshmem_uint64_wait_until_any(my_send_status, MSG_IN_FLIGHT_MAX,
      nullptr, NVSHMEM_CMP_EQ, SIGNAL_FREE);
  nvshmemx_signal_op(my_send_status + msg_idx, SIGNAL_USED, NVSHMEM_SIGNAL_SET, c_my_pe);

  // Send alloy of source buffer (offset & size)
  signal_offset = MSG_IN_FLIGHT_MAX * c_my_pe;
  uint64_t* my_recv_alloy = recv_alloy + signal_offset;
  alloy src_alloy(offset, msg_size);
  nvshmemx_signal_op(my_recv_alloy + msg_idx, src_alloy.data, NVSHMEM_SIGNAL_SET, dst_pe);
#ifdef DEBUG
  assert(msg_idx != SIZE_MAX);
  printf("PE %d sending message request: src addr %p, dst PE %d, idx %llu, size %llu\n",
      c_my_pe, (void*)mbuf->addr(offset), dst_pe, msg_idx, msg_size);
#endif

  // Store source alloy for later cleanup
  send_alloy[signal_offset + msg_idx] = src_alloy.data;
}

__device__ void charm::send_dummy_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  size_t offset;
  envelope* env = create_envelope(msgtype::dummy, msg_size, &offset);

  send_msg(offset, msg_size, dst_pe);
}

__device__ void charm::send_reg_msg(int chare_id, int chare_idx, int ep_id,
                                    void* buf, size_t payload_size, int dst_pe) {
  size_t msg_size = envelope::alloc_size(sizeof(regular_msg) + payload_size);
  size_t offset;
  envelope* env = create_envelope(msgtype::regular, msg_size, &offset);

  regular_msg* msg = new ((char*)env + sizeof(envelope)) regular_msg(chare_id, chare_idx, ep_id);

  // Fill in payload (from regular GPU memory to NVSHMEM symmetric memory)
  if (payload_size > 0) {
    memcpy((char*)msg + sizeof(regular_msg), buf, payload_size);
  }

  send_msg(offset, msg_size, dst_pe);
}

__device__ void charm::send_begin_term_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  size_t offset;
  envelope* env = create_envelope(msgtype::begin_terminate, msg_size, &offset);

  send_msg(offset, msg_size, dst_pe);
}

__device__ void charm::send_do_term_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  size_t offset;
  envelope* env = create_envelope(msgtype::do_terminate, msg_size, &offset);

  send_msg(offset, msg_size, dst_pe);
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

__device__ __forceinline__ void recv_msg(min_heap& addr_heap, bool& begin_term_flag,
    bool &do_term_flag) {
  // Check if there are any message requests
  size_t count = nvshmem_uint64_test_some(recv_alloy, MSG_IN_FLIGHT_MAX * c_n_pes,
      recv_alloy_idx, nullptr, NVSHMEM_CMP_GT, 0);
  if (count > 0) {
    for (size_t i = 0; i < count; i++) {
      // Obtain information about this message request
      size_t msg_idx = recv_alloy_idx[i];
      uint64_t data = nvshmem_signal_fetch(recv_alloy + msg_idx);
      alloy src_alloy(data);
      size_t src_offset = src_alloy.offset();
      size_t src_size = src_alloy.size();
      int src_pe = msg_idx / MSG_IN_FLIGHT_MAX;
      msg_idx -= MSG_IN_FLIGHT_MAX * src_pe;

      // Reserve space for incoming message
      ringbuf_off_t mret = spsc_ringbuf_acquire(mbuf, src_size);
      assert(mret != -1 && mret < mbuf_size);

      // Perform a get operation to fetch the message
      // TODO: Make asynchronous
      nvshmem_char_get((char*)mbuf->addr(mret), (char*)mbuf->addr(src_offset), src_size, src_pe);
#ifdef DEBUG
      printf("PE %d receiving message: src addr %p, dst addr %p, src PE %d, idx %llu, size %llu\n",
          c_my_pe, (void*)mbuf->addr(src_offset), (void*)mbuf->addr(mret), src_pe, msg_idx, src_size);
#endif

      // Process message
      next_msg(mbuf->addr(mret), begin_term_flag, do_term_flag);

      // Clear message request
      nvshmemx_signal_op(recv_alloy + MSG_IN_FLIGHT_MAX * src_pe + msg_idx,
          SIGNAL_FREE, NVSHMEM_SIGNAL_SET, c_my_pe);

      // Store alloy to be cleared from memory
      addr_heap.push(src_alloy);
#ifdef DEBUG
      printf("PE %d flagging message for cleanup: dst addr %p, src PE %d, idx %llu, size %llu\n",
          c_my_pe, mbuf->addr(mret), src_pe, msg_idx, src_size);
#endif

      // Notify sender that message is ready for cleanup
      nvshmemx_signal_op(send_status + MSG_IN_FLIGHT_MAX * c_my_pe + msg_idx,
          SIGNAL_CLUP, NVSHMEM_SIGNAL_SET, src_pe);
    }

    // Reset indices array for next use
    memset(recv_alloy_idx, 0, MSG_IN_FLIGHT_MAX * c_n_pes * sizeof(size_t));
  }

  // Check for messages that have been delivered to the destination PE
  count = nvshmem_uint64_test_some(send_status, MSG_IN_FLIGHT_MAX * c_n_pes,
      send_status_idx, nullptr, NVSHMEM_CMP_EQ, SIGNAL_CLUP);
  if (count > 0) {
    for (size_t i = 0; i < count; i++) {
      size_t msg_idx = send_status_idx[i];

      // Store alloy to be cleared from memory
      alloy src_alloy(send_alloy[msg_idx]);
      addr_heap.push(src_alloy);
#ifdef DEBUG
      printf("PE %d flagging message for cleanup: src offset %llu, size %llu, "
          "dst PE %llu, idx %llu\n", c_my_pe, src_alloy.offset(), src_alloy.size(),
          msg_idx / MSG_IN_FLIGHT_MAX, msg_idx % MSG_IN_FLIGHT_MAX);
#endif

      // Reset signal to SIGNAL_FREE
      nvshmemx_signal_op(send_status + msg_idx, SIGNAL_FREE,
          NVSHMEM_SIGNAL_SET, c_my_pe);
    }

    // Reset indices array for next use
    memset(send_status_idx, 0, MSG_IN_FLIGHT_MAX * c_n_pes * sizeof(size_t));
  }
}

__global__ void charm::scheduler(int argc, char** argv, size_t* argvs) {
  if (!blockIdx.x && !threadIdx.x) {
    min_heap addr_heap(heap_buf, heap_buf_size / sizeof(uint64_t));
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
      recv_msg(addr_heap, begin_term_flag, do_term_flag);
    } while (!do_term_flag);

#if DEBUG
    printf("PE %d terminating...\n", c_my_pe);
#endif
  }
}
