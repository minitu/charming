#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "ringbuf.h"
#include "composite.h"
#include "heap.h"
#include "util.h"

// Maximum number of messages that are allowed to be in flight per pair of PEs
#define REMOTE_MSG_COUNT_MAX 4
// Maximum number of messages that can be stored in the local message queue
#define LOCAL_MSG_COUNT_MAX 128

using namespace charm;

extern cudaStream_t stream;

// GPU constant memory
extern __constant__ int c_my_pe;
extern __constant__ int c_n_pes;
extern __constant__ int c_my_pe_node;
extern __constant__ int c_n_pes_node;
extern __constant__ int c_n_nodes;

// GPU global memory
__device__ ringbuf_t* mbuf;
__device__ size_t mbuf_size;
__device__ uint64_t* send_status;
__device__ uint64_t* recv_remote_comp;
__device__ compbuf_t* recv_local_comp;
__device__ uint64_t* send_comp;
__device__ size_t* send_status_idx;
__device__ size_t* recv_remote_comp_idx;
__device__ composite_t* heap_buf;
__device__ size_t heap_buf_size;

// Host memory
uint64_t* h_send_status;
uint64_t* h_recv_remote_comp;
uint64_t* h_send_comp;
uint64_t* h_send_status_idx;
uint64_t* h_recv_remote_comp_idx;
composite_t* h_heap_buf;
ringbuf_t* h_mbuf;
ringbuf_t* h_mbuf_d;
compbuf_t* h_recv_local_comp;
compbuf_t* h_recv_local_comp_d;

enum {
  SIGNAL_FREE = 0,
  SIGNAL_USED = 1,
  SIGNAL_CLUP = 2
};

void charm::comm_init_host(int n_pes) {
  // Allocate message buffer
  size_t h_mbuf_size = 1073741824; // TODO
  cudaMallocHost(&h_mbuf, sizeof(ringbuf_t));
  cudaMalloc(&h_mbuf_d, sizeof(ringbuf_t));
  assert(h_mbuf && h_mbuf_d);
  h_mbuf->init(h_mbuf_size);

  // Allocate data structures
  size_t h_status_size = REMOTE_MSG_COUNT_MAX * n_pes * sizeof(uint64_t);
  h_send_status = (uint64_t*)nvshmem_malloc(h_status_size);
  size_t h_remote_comp_size = REMOTE_MSG_COUNT_MAX * n_pes * sizeof(uint64_t);
  h_recv_remote_comp = (uint64_t*)nvshmem_malloc(h_remote_comp_size);
  cudaMallocHost(&h_recv_local_comp, sizeof(compbuf_t));
  cudaMalloc(&h_recv_local_comp_d, sizeof(compbuf_t));
  assert(h_recv_local_comp && h_recv_local_comp_d);
  h_recv_local_comp->init(LOCAL_MSG_COUNT_MAX);
  cudaMalloc(&h_send_comp, h_remote_comp_size);
  size_t h_idx_size = REMOTE_MSG_COUNT_MAX * n_pes * sizeof(size_t);
  cudaMalloc(&h_send_status_idx, h_idx_size);
  cudaMalloc(&h_recv_remote_comp_idx, h_idx_size);
  size_t h_heap_buf_size = REMOTE_MSG_COUNT_MAX * n_pes * 2 * sizeof(composite_t);
  cudaMalloc(&h_heap_buf, h_heap_buf_size);
  assert(h_send_status && h_recv_remote_comp && h_send_comp && h_send_status_idx
      && h_recv_remote_comp_idx && h_heap_buf);
  cuda_check_error();

  // Prepare data structures
  cudaMemcpyAsync(h_mbuf_d, h_mbuf, sizeof(ringbuf_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf_d, sizeof(ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);

  cudaMemcpyToSymbolAsync(send_status, &h_send_status, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_remote_comp, &h_recv_remote_comp, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(h_recv_local_comp_d, h_recv_local_comp, sizeof(compbuf_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_local_comp, &h_recv_local_comp_d, sizeof(compbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_comp, &h_send_comp, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_status_idx, &h_send_status_idx, sizeof(size_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_remote_comp_idx, &h_recv_remote_comp_idx, sizeof(size_t*), 0, cudaMemcpyHostToDevice, stream);

  cudaMemsetAsync(h_send_status, 0, h_status_size, stream);
  cudaMemsetAsync(h_recv_remote_comp, 0, h_remote_comp_size, stream);
  cudaMemsetAsync(h_send_comp, 0, h_remote_comp_size, stream);
  cudaMemsetAsync(h_send_status_idx, 0, h_idx_size, stream);
  cudaMemsetAsync(h_recv_remote_comp_idx, 0, h_idx_size, stream);

  cudaMemcpyToSymbolAsync(heap_buf, &h_heap_buf, sizeof(composite_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(heap_buf_size, &h_heap_buf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(h_heap_buf, 0, h_heap_buf_size, stream);
  cuda_check_error();
}

void charm::comm_fini_host(int n_pes) {
  nvshmem_free(h_send_status);
  nvshmem_free(h_recv_remote_comp);

  cudaFree(h_send_comp);
  cudaFree(h_send_status_idx);
  cudaFree(h_recv_remote_comp_idx);
  cudaFree(h_heap_buf);

  h_mbuf->fini();
  cudaFreeHost(h_mbuf);
  cudaFree(h_mbuf_d);

  h_recv_local_comp->fini();
  cudaFreeHost(h_recv_local_comp);
  cudaFree(h_recv_local_comp_d);
}

__device__ void charm::comm::init() {
  addr_heap.init(heap_buf, heap_buf_size / sizeof(uint64_t));
  begin_term_flag = false;
  do_term_flag = false;
}

__device__ void charm::comm::process_local() {
  // Check local composite queue
  if (recv_local_comp->count > 0) {
    uint64_t data = *(recv_local_comp->addr(recv_local_comp->read));
    composite_t src_composite(data);
    size_t src_offset = src_composite.offset();
    size_t msg_size = src_composite.size();
    void* addr = mbuf->addr(src_offset);
    PDEBUG("PE %d receiving local message: offset %llu, size %llu\n",
        c_my_pe, src_offset, msg_size);

    // Process message
    msgtype type = process_msg(addr, nullptr, begin_term_flag, do_term_flag);

    // Release composite from local queue
    recv_local_comp->release();

#ifndef NO_CLEANUP
    if (type != msgtype::user) {
      // Store composite to be cleared from memory
      addr_heap.push(src_composite);
      PDEBUG("PE %d flagging local message for cleanup: offset %llu, "
          "size %llu\n", c_my_pe, src_offset, msg_size);
    }
#endif // !NO_CLEANUP
  }
}

__device__ void charm::comm::process_remote() {
  // Check if there are any message requests
  size_t count = nvshmem_uint64_test_some(recv_remote_comp, REMOTE_MSG_COUNT_MAX * c_n_pes,
      recv_remote_comp_idx, nullptr, NVSHMEM_CMP_GT, 0);
  if (count > 0) {
    for (size_t i = 0; i < count; i++) {
      // Obtain information about this message request
      size_t msg_idx = recv_remote_comp_idx[i];
      uint64_t data = nvshmem_signal_fetch(recv_remote_comp + msg_idx);
      composite_t src_composite(data);
      size_t src_offset = src_composite.offset();
      size_t msg_size = src_composite.size();
      int src_pe = msg_idx / REMOTE_MSG_COUNT_MAX;
      msg_idx -= REMOTE_MSG_COUNT_MAX * src_pe;

      // Reserve space for incoming message
      ringbuf_off_t dst_offset = mbuf->acquire(msg_size);
      if (dst_offset == -1) {
        PERROR("PE %d: Not enough space in message buffer\n", c_my_pe);
        assert(false);
      }
      PDEBUG("PE %d acquired message: offset %llu, size %llu\n",
          c_my_pe, dst_offset, msg_size);

      // Perform a get operation to fetch the message
      // TODO: Make asynchronous
      void* dst_addr = mbuf->addr(dst_offset);
      nvshmem_char_get((char*)dst_addr, (char*)mbuf->addr(src_offset),
          msg_size, src_pe);
      PDEBUG("PE %d receiving message: src offset %llu, dst offset %llu, "
          "size %llu, src PE %d, idx %llu\n", c_my_pe, src_offset, dst_offset,
          msg_size, src_pe, msg_idx);

      // Process message
      msgtype type = process_msg(dst_addr, nullptr, begin_term_flag, do_term_flag);

      // Clear message request
      nvshmemx_signal_op(recv_remote_comp + REMOTE_MSG_COUNT_MAX * src_pe + msg_idx,
          SIGNAL_FREE, NVSHMEM_SIGNAL_SET, c_my_pe);

#ifndef NO_CLEANUP
      if (type != msgtype::user) {
        // Store composite to be cleared from memory
        composite_t dst_composite(dst_offset, msg_size);
        addr_heap.push(dst_composite);
        PDEBUG("PE %d flagging received message for cleanup: offset %llu, "
            "size %llu, src PE %d, idx %llu\n", c_my_pe, dst_composite.offset(),
            dst_composite.size(), src_pe, msg_idx);
      }

      // Notify sender that message is ready for cleanup
      int signal = (type == msgtype::user) ? SIGNAL_FREE : SIGNAL_CLUP;
      nvshmemx_signal_op(send_status + REMOTE_MSG_COUNT_MAX * c_my_pe + msg_idx,
          signal, NVSHMEM_SIGNAL_SET, src_pe);
#else
      // Notify sender that message has been delivered
      nvshmemx_signal_op(send_status + REMOTE_MSG_COUNT_MAX * c_my_pe + msg_idx,
          SIGNAL_FREE, NVSHMEM_SIGNAL_SET, src_pe);
#endif // !NO_CLEANUP
    }

    // Reset indices array for next use
    for (int i = 0; i < REMOTE_MSG_COUNT_MAX * c_n_pes; i++) {
      recv_remote_comp_idx[i] = 0;
    }
    //memset(recv_remote_comp_idx, 0, REMOTE_MSG_COUNT_MAX * c_n_pes * sizeof(size_t));
  }
}

__device__ void charm::comm::cleanup() {
#ifndef NO_CLEANUP
  // Check for messages that have been delivered to the destination PE
  size_t count = nvshmem_uint64_test_some(send_status, REMOTE_MSG_COUNT_MAX * c_n_pes,
      send_status_idx, nullptr, NVSHMEM_CMP_EQ, SIGNAL_CLUP);
  if (count > 0) {
    for (size_t i = 0; i < count; i++) {
      size_t msg_idx = send_status_idx[i];

      // Store composite to be cleared from memory
      composite_t src_composite(send_comp[msg_idx]);
      addr_heap.push(src_composite);
      PDEBUG("PE %d flagging sent message for cleanup: offset %llu, size %llu, "
          "dst PE %llu, idx %llu\n", c_my_pe, src_composite.offset(),
          src_composite.size(), msg_idx / REMOTE_MSG_COUNT_MAX,
          msg_idx % REMOTE_MSG_COUNT_MAX);

      // Reset signal to SIGNAL_FREE
      nvshmemx_signal_op(send_status + msg_idx, SIGNAL_FREE,
          NVSHMEM_SIGNAL_SET, c_my_pe);
    }

    // Reset indices array for next use
    memset(send_status_idx, 0, REMOTE_MSG_COUNT_MAX * c_n_pes * sizeof(size_t));
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
      PDEBUG("PE %d releasing message: offset %llu, size %llu\n",
          c_my_pe, clup_offset, clup_size);
    } else break;
  }
#endif // !NO_CLEANUP
}

__device__ void charm::message::alloc(size_t size) {
  size_t msg_size = envelope::alloc_size(sizeof(regular_msg) + size);
  env = create_envelope(msgtype::user, msg_size, &offset);
}

__device__ void charm::message::free() {
  // TODO
}

__device__ envelope* charm::create_envelope(msgtype type, size_t msg_size, size_t* offset) {
  // Reserve space for this message in message buffer
  ringbuf_off_t mbuf_off = mbuf->acquire(msg_size);
  if (mbuf_off == -1) {
    PERROR("PE %d: Not enough space in message buffer\n", c_my_pe);
    assert(false);
  }
  PDEBUG("PE %d acquired message: offset %llu, size %llu\n", c_my_pe, mbuf_off, msg_size);
  *offset = mbuf_off;

  // Create envelope
  return new (mbuf->addr(mbuf_off)) envelope(type, msg_size, c_my_pe);
}

__device__ void charm::send_msg(envelope* env, size_t offset, size_t msg_size, int dst_pe) {
  // Create composite using offset and size of source buffer
  composite_t src_composite(offset, msg_size);

  if (dst_pe == c_my_pe) {
    // Sending message to itself
    // Acquire space in local composite queue and store composite
    compbuf_off_t local_offset = recv_local_comp->acquire();
    if (local_offset < 0) {
      PERROR("Out of space in local composite queue\n");
      assert(false);
    }
    *(composite_t*)recv_local_comp->addr(local_offset) = src_composite;
    PDEBUG("PE %d sending local message: offset %llu, local %lld, size %llu\n",
        c_my_pe, offset, local_offset, msg_size);
  } else {
    // Sending message to a different PE
    // Obtain a message index for the target PE and set the corresponding used signal
    size_t send_offset = REMOTE_MSG_COUNT_MAX * dst_pe;
    uint64_t* my_send_status = send_status + send_offset;
    size_t msg_idx = nvshmem_uint64_wait_until_any(my_send_status, REMOTE_MSG_COUNT_MAX,
        nullptr, NVSHMEM_CMP_EQ, SIGNAL_FREE);
    nvshmemx_signal_op(my_send_status + msg_idx, SIGNAL_USED,
        NVSHMEM_SIGNAL_SET, c_my_pe);

    // Send composite
    size_t recv_offset = REMOTE_MSG_COUNT_MAX * c_my_pe;
    uint64_t* my_recv_remote_comp = recv_remote_comp + recv_offset;
    nvshmemx_signal_op(my_recv_remote_comp + msg_idx, src_composite.data,
        NVSHMEM_SIGNAL_SET, dst_pe);
    assert(msg_idx != SIZE_MAX);
    PDEBUG("PE %d sending message request: offset %llu, size %llu, dst PE %d, idx %llu\n",
        c_my_pe, offset, msg_size, dst_pe, msg_idx);

#ifndef NO_CLEANUP
    // Store source composite for later cleanup
    send_comp[send_offset + msg_idx] = src_composite.data;
#endif
  }
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

  send_msg(env, offset, msg_size, dst_pe);
}

__device__ void charm::send_user_msg(int chare_id, int chare_idx, int ep_id,
    const message& msg, int dst_pe) {
  // Set regular message fields using placement new
  envelope* env = msg.env;
  new ((char*)env + sizeof(envelope)) regular_msg(chare_id, chare_idx, ep_id);

  send_msg(env, msg.offset, env->size, dst_pe);
}

__device__ void charm::send_user_msg(int chare_id, int chare_idx, int ep_id,
    const message& msg, size_t payload_size, int dst_pe) {
  // Set regular message fields using placement new
  envelope* env = msg.env;
  new ((char*)env + sizeof(envelope)) regular_msg(chare_id, chare_idx, ep_id);

  // Message size can be smaller than allocated
  size_t msg_size = envelope::alloc_size(sizeof(regular_msg) + payload_size);

  send_msg(env, msg.offset, msg_size, dst_pe);
}

__device__ void charm::send_begin_term_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  size_t offset;
  envelope* env = create_envelope(msgtype::begin_terminate, msg_size, &offset);

  send_msg(env, offset, msg_size, dst_pe);
}

__device__ void charm::send_do_term_msg(int dst_pe) {
  size_t msg_size = envelope::alloc_size(0);
  size_t offset;
  envelope* env = create_envelope(msgtype::do_terminate, msg_size, &offset);

  send_msg(env, offset, msg_size, dst_pe);
}

