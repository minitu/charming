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

extern __device__ ringbuf_t* mbuf;
extern __device__ size_t mbuf_size;
extern __device__ uint64_t* send_status;
extern __device__ uint64_t* recv_composite;
extern __device__ uint64_t* send_composite;
extern __device__ size_t* send_status_idx;
extern __device__ size_t* recv_composite_idx;
extern __device__ composite_t* heap_buf;
extern __device__ size_t heap_buf_size;

extern __device__ chare_proxy_base* chare_proxies[];

enum {
  SIGNAL_FREE = 0,
  SIGNAL_USED = 1,
  SIGNAL_CLUP = 2
};

__device__ envelope* charm::create_envelope(msgtype type, size_t msg_size, size_t* offset) {
  // Reserve space for this message in message buffer
  ringbuf_off_t mbuf_off = mbuf->acquire(msg_size);
  if (mbuf_off == -1) {
    printf("PE %d error: Not enough space in message buffer\n", c_my_pe);
    assert(false);
  }
#ifdef DEBUG
  printf("PE %d acquired message: offset %llu, size %llu\n", c_my_pe, mbuf_off, msg_size);
#endif
  *offset = mbuf_off;

  // Create envelope
  return new (mbuf->addr(mbuf_off)) envelope(type, msg_size, c_my_pe);
}

__device__ void charm::send_msg(size_t offset, size_t msg_size, int dst_pe) {
  // Obtain a message index for the target PE and set the corresponding used signal
  size_t send_offset = MSG_IN_FLIGHT_MAX * dst_pe;
  uint64_t* my_send_status = send_status + send_offset;
  size_t msg_idx = nvshmem_uint64_wait_until_any(my_send_status, MSG_IN_FLIGHT_MAX,
      nullptr, NVSHMEM_CMP_EQ, SIGNAL_FREE);
  nvshmemx_signal_op(my_send_status + msg_idx, SIGNAL_USED,
      NVSHMEM_SIGNAL_SET, c_my_pe);

  // Send composite of source buffer (offset & size)
  size_t recv_offset = MSG_IN_FLIGHT_MAX * c_my_pe;
  uint64_t* my_recv_composite = recv_composite + recv_offset;
  composite_t src_composite(offset, msg_size);
  nvshmemx_signal_op(my_recv_composite + msg_idx, src_composite.data,
      NVSHMEM_SIGNAL_SET, dst_pe);
#ifdef DEBUG
  assert(msg_idx != SIZE_MAX);
  printf("PE %d sending message request: offset %llu, size %llu, dst PE %d, idx %llu\n",
      c_my_pe, offset, msg_size, dst_pe, msg_idx);
#endif

  // Store source composite for later cleanup
  send_composite[send_offset + msg_idx] = src_composite.data;
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

  regular_msg* msg = new ((char*)env + sizeof(envelope)) regular_msg(chare_id,
      chare_idx, ep_id);

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
         c_my_pe, env->type, env->size, env->src_pe);
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
    printf("PE %d creation msg chare ID %d, n_local %d, n_total %d, "
        "start idx %d, end idx %d\n", c_my_pe, msg->chare_id, msg->n_local,
        msg->n_total, msg->start_idx, msg->end_idx);
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
    printf("PE %d regular msg chare ID %d chare idx %d EP ID %d\n", c_my_pe,
        msg->chare_id, msg->chare_idx, msg->ep_id);
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
    printf("PE %d begin terminate msg\n", c_my_pe);
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
    printf("PE %d do terminate msg\n", c_my_pe);
#endif
    do_term_flag = true;
  }

  return env->size;
}

__device__ __forceinline__ void recv_msg(min_heap& addr_heap, bool& begin_term_flag,
    bool &do_term_flag) {
  // Check if there are any message requests
  size_t count = nvshmem_uint64_test_some(recv_composite, MSG_IN_FLIGHT_MAX * c_n_pes,
      recv_composite_idx, nullptr, NVSHMEM_CMP_GT, 0);
  if (count > 0) {
    for (size_t i = 0; i < count; i++) {
      // Obtain information about this message request
      size_t msg_idx = recv_composite_idx[i];
      uint64_t data = nvshmem_signal_fetch(recv_composite + msg_idx);
      composite_t src_composite(data);
      size_t src_offset = src_composite.offset();
      size_t msg_size = src_composite.size();
      int src_pe = msg_idx / MSG_IN_FLIGHT_MAX;
      msg_idx -= MSG_IN_FLIGHT_MAX * src_pe;

      // Reserve space for incoming message
      ringbuf_off_t dst_offset = mbuf->acquire(msg_size);
      if (dst_offset == -1) {
        printf("PE %d error: Not enough space in message buffer\n", c_my_pe);
        assert(false);
      }
#ifdef DEBUG
  printf("PE %d acquired message: offset %llu, size %llu\n",
      c_my_pe, dst_offset, msg_size);
#endif

      // Perform a get operation to fetch the message
      // TODO: Make asynchronous
      void* dst_addr = mbuf->addr(dst_offset);
      nvshmem_char_get((char*)dst_addr, (char*)mbuf->addr(src_offset),
          msg_size, src_pe);
#ifdef DEBUG
      printf("PE %d receiving message: src offset %llu, dst offset %llu, "
          "size %llu, src PE %d, idx %llu\n", c_my_pe, src_offset, dst_offset,
          msg_size, src_pe, msg_idx);
#endif

      // Process message
      next_msg(dst_addr, begin_term_flag, do_term_flag);

      // Clear message request
      nvshmemx_signal_op(recv_composite + MSG_IN_FLIGHT_MAX * src_pe + msg_idx,
          SIGNAL_FREE, NVSHMEM_SIGNAL_SET, c_my_pe);

      // Store composite to be cleared from memory
      composite_t dst_composite(dst_offset, msg_size);
      addr_heap.push(dst_composite);
#ifdef DEBUG
      printf("PE %d flagging received message for cleanup: offset %llu, "
          "size %llu, src PE %d, idx %llu\n", c_my_pe, dst_composite.offset(),
          dst_composite.size(), src_pe, msg_idx);
#endif

      // Notify sender that message is ready for cleanup
      nvshmemx_signal_op(send_status + MSG_IN_FLIGHT_MAX * c_my_pe + msg_idx,
          SIGNAL_CLUP, NVSHMEM_SIGNAL_SET, src_pe);
    }

    // Reset indices array for next use
    memset(recv_composite_idx, 0, MSG_IN_FLIGHT_MAX * c_n_pes * sizeof(size_t));
  }

  // Check for messages that have been delivered to the destination PE
  count = nvshmem_uint64_test_some(send_status, MSG_IN_FLIGHT_MAX * c_n_pes,
      send_status_idx, nullptr, NVSHMEM_CMP_EQ, SIGNAL_CLUP);
  if (count > 0) {
    for (size_t i = 0; i < count; i++) {
      size_t msg_idx = send_status_idx[i];

      // Store composite to be cleared from memory
      composite_t src_composite(send_composite[msg_idx]);
      addr_heap.push(src_composite);
#ifdef DEBUG
      printf("PE %d flagging sent message for cleanup: offset %llu, size %llu, "
          "dst PE %llu, idx %llu\n", c_my_pe, src_composite.offset(),
          src_composite.size(), msg_idx / MSG_IN_FLIGHT_MAX,
          msg_idx % MSG_IN_FLIGHT_MAX);
#endif

      // Reset signal to SIGNAL_FREE
      nvshmemx_signal_op(send_status + msg_idx, SIGNAL_FREE,
          NVSHMEM_SIGNAL_SET, c_my_pe);
    }

    // Reset indices array for next use
    memset(send_status_idx, 0, MSG_IN_FLIGHT_MAX * c_n_pes * sizeof(size_t));
  }

  // Clean up messages
  while (true) {
    composite_t comp = addr_heap.top();
    if (comp.data == UINT64_MAX) break;

    size_t clup_offset = comp.offset();
    size_t clup_size = comp.size();
    if (clup_offset == mbuf->read && clup_size > 0) {
      mbuf->release(clup_size);
      addr_heap.pop();
#ifdef DEBUG
      printf("PE %d releasing message: offset %llu, size %llu\n",
          c_my_pe, clup_offset, clup_size);
#endif
    } else break;
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

    nvshmem_barrier_all();

    if (c_my_pe == 0) {
      // Execute user's main function
      main(argc, argv, argvs);
    }

    // FIXME: Is this barrier necessary?
    nvshmem_barrier_all();

    // Receive messages and terminate
    do {
      recv_msg(addr_heap, begin_term_flag, do_term_flag);
    } while (!do_term_flag);

#if DEBUG
    printf("PE %d terminating...\n", c_my_pe);
#endif
  }
}
