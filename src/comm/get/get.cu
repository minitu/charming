#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "common.h"
#include "kernel.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "ringbuf.h"
#include "composite.h"
#include "heap.h"
#include "util.h"

// Use custom thread block extensions to NVSHMEM
#define NVSHMEM_BLOCK_IMPL

/*
// Maximum number of messages that are allowed to be in flight per pair of PEs
#define REMOTE_MSG_COUNT_MAX 4
// Maximum number of messages that can be stored in the local message queue
#define LOCAL_MSG_COUNT_MAX 128
*/
#define LOCAL_MAX 4

using namespace charm;

typedef unsigned long long int atomic64_t;

extern cudaStream_t stream;

// GPU constant memory
extern __constant__ int c_n_sms;
extern __constant__ int c_my_dev;
extern __constant__ int c_my_dev_node;
extern __constant__ int c_n_devs;
extern __constant__ int c_n_devs_node;
extern __constant__ int c_n_nodes;
extern __constant__ int c_n_pes;
extern __constant__ int c_n_pes_node;

// Managed memory (actual data may reside in GPU global memory)
/*
__device__ __managed__ ringbuf_t* mbuf; // Managed
__device__ __managed__ uint64_t* send_status; // NVSHMEM
__device__ __managed__ uint64_t* recv_remote_comp; // NVSHMEM
__device__ __managed__ compbuf_t* recv_local_comp; // Managed
__device__ __managed__ uint64_t* send_comp; // Global
__device__ __managed__ size_t* send_status_idx; // Global
__device__ __managed__ size_t* recv_remote_comp_idx; // Global
__device__ __managed__ composite_t* heap_buf; // Global
__device__ __managed__ size_t heap_buf_size;
*/
__managed__ void* nvshmem_buf; // NVSHMEM
__managed__ ringbuf_t* mbuf; // Managed
__managed__ volatile int* send_status; // Global
__managed__ atomic64_t* send_comp; // Global
__managed__ volatile atomic64_t* recv_comp; // Global
__managed__ composite_t* heap_buf; // Global

// GPU shared memory
extern __shared__ uint64_t s_mem[];

enum {
  SIGNAL_FREE = 0,
  SIGNAL_USED = 1,
  SIGNAL_CLUP = 2
};

void charm::comm_init_host(int n_pes, int n_sms) {
  // Allocate NVSHMEM message buffer
  constexpr size_t mbuf_size = 8388608; // TODO: 8MB per SM
  nvshmem_buf = nvshmem_malloc(mbuf_size * n_sms);
  assert(nvshmem_buf);
  cudaMallocManaged(&mbuf, sizeof(ringbuf_t) * n_sms);
  assert(mbuf);
  ringbuf_t* cur_mbuf = mbuf;
  for (int i = 0; i < n_sms; i++) {
    cur_mbuf->init(nvshmem_buf, mbuf_size, i);
    cur_mbuf++;
  }

  // Allocate data structures
  size_t count = LOCAL_MAX * n_sms * n_sms;
  size_t status_size = sizeof(int) * count;
  size_t comp_size = sizeof(atomic64_t) * count;
  size_t heap_size = sizeof(composite_t) * count * 2;
  cudaMalloc(&send_status, status_size);
  cudaMalloc(&send_comp, comp_size);
  cudaMalloc(&recv_comp, comp_size);
  cudaMalloc(&heap_buf, heap_size);
  assert(send_status && send_comp && recv_comp && heap_buf);

  // Clear data structures
  cudaMemsetAsync((void*)send_status, 0, status_size, stream);
  cudaMemsetAsync(send_comp, 0, comp_size, stream);
  cudaMemsetAsync((void*)recv_comp, 0, comp_size, stream);
  cudaMemsetAsync(heap_buf, 0, heap_size, stream);
  cudaStreamSynchronize(stream);

  /*
  // Allocate message buffer
  size_t mbuf_size = 8388608; // TODO
  cudaMallocManaged(&mbuf, sizeof(ringbuf_t) * n_sms);
  assert(mbuf);
  ringbuf_t* cur_mbuf = mbuf;
  for (int i = 0; i < n_sms; i++) {
    cur_mbuf->init(mbuf_size);
    cur_mbuf++;
  }

  // Allocate and prepare data structures
  size_t status_size = REMOTE_MSG_COUNT_MAX * n_pes * n_sms * sizeof(uint64_t);
  send_status = (uint64_t*)nvshmem_malloc(status_size);
  assert(send_status);

  size_t remote_comp_size = status_size;
  recv_remote_comp = (uint64_t*)nvshmem_malloc(remote_comp_size);
  assert(recv_remote_comp);

  cudaMallocManaged(&recv_local_comp, sizeof(compbuf_t) * n_sms);
  assert(recv_local_comp);
  compbuf_t* cur_comp = recv_local_comp;
  for (int i = 0; i < n_sms; i++) {
    cur_comp->init(LOCAL_MSG_COUNT_MAX);
    cur_comp++;
  }

  cudaMalloc(&send_comp, remote_comp_size);
  assert(send_comp);

  size_t idx_size = REMOTE_MSG_COUNT_MAX * n_pes * n_sms * sizeof(size_t);
  cudaMalloc(&send_status_idx, idx_size);
  cudaMalloc(&recv_remote_comp_idx, idx_size);
  assert(send_status_idx && recv_remote_comp_idx);

  heap_buf_size = REMOTE_MSG_COUNT_MAX * n_pes * 2 * n_sms * sizeof(composite_t);
  cudaMalloc(&heap_buf, heap_buf_size);
  assert(heap_buf);

  // Clear data structures
  cudaMemsetAsync(send_status, 0, status_size, stream);
  cudaMemsetAsync(recv_remote_comp, 0, remote_comp_size, stream);
  cudaMemsetAsync(send_comp, 0, remote_comp_size, stream);
  cudaMemsetAsync(send_status_idx, 0, idx_size, stream);
  cudaMemsetAsync(recv_remote_comp_idx, 0, idx_size, stream);
  cudaMemsetAsync(heap_buf, 0, heap_buf_size, stream);

  cudaStreamSynchronize(stream);
  cuda_check_error();
  */
}

void charm::comm_fini_host(int n_pes, int n_sms) {
  nvshmem_free(nvshmem_buf);
  cudaFree(mbuf);
  cudaFree((void*)send_status);
  cudaFree(send_comp);
  cudaFree((void*)recv_comp);
  cudaFree(heap_buf);

  /*
  cudaFree(send_comp);
  cudaFree(send_status_idx);
  cudaFree(recv_remote_comp_idx);
  cudaFree(heap_buf);

  compbuf_t* cur_comp = recv_local_comp;
  for (int i = 0; i < n_sms; i++) {
    cur_comp->fini();
    cur_comp++;
  }
  cudaFree(recv_local_comp);

  nvshmem_free(send_status);
  nvshmem_free(recv_remote_comp);

  ringbuf_t* cur_mbuf = mbuf;
  for (int i = 0; i < n_sms; i++) {
    cur_mbuf->fini();
    cur_mbuf++;
  }
  cudaFree(mbuf);
  */
}

__device__ void charm::comm::init() {
  int comp_count = LOCAL_MAX * c_n_sms * 2;
  composite_t* my_heap_buf = heap_buf + comp_count * blockIdx.x;
  addr_heap.init(my_heap_buf, comp_count);

  begin_term_flag = false;
  do_term_flag = false;
}

__device__ __forceinline__ int find_signal_single(volatile int* status,
    int count, int old_val, int new_val, bool loop) {
  int idx = INT_MAX;

  // Look for desired signal
  do {
    for (int i = 0; i < count; i++) {
      if (status[i] == old_val) {
        idx = i;
      }
      __threadfence();

      if (idx != INT_MAX) break;
    }
  } while (loop && idx == INT_MAX);

  // Update signal if necessary
  if (idx != INT_MAX && old_val != new_val) {
    int ret = atomicCAS((int*)&status[idx], old_val, new_val);
    assert(ret == old_val);
  }

  return idx;
}

__device__ __forceinline__ int find_signal_block(volatile int* status,
    int count, int old_val, int new_val, bool loop) {
  __shared__ volatile int idx;
  if (threadIdx.x == 0) idx = INT_MAX;
  __syncthreads();

  // Look for desired signal
  do {
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
      if (status[i] == old_val) {
        atomicMin_block((int*)&idx, i);
      }
      __threadfence();
    }
  } while (loop && idx == INT_MAX);
  __syncthreads();

  // Update signal if necessary
  if (idx != INT_MAX && old_val != new_val && threadIdx.x == 0) {
    int ret = atomicCAS((int*)&status[idx], old_val, new_val);
    assert(ret == old_val);
  }
  __syncthreads();

  return idx;
}

__device__ __forceinline__ atomic64_t find_msg_single(volatile atomic64_t* comps, int& idx) {
  idx = INT_MAX;
  atomic64_t comp = 0;

  // Look for a valid message (traverse once)
  for (int i = 0; i < LOCAL_MAX * c_n_sms; i++) {
    comp = comps[i];
    __threadfence();

    // If a message is found, reset message address to zero
    if (comp) {
      idx = i;
      atomic64_t ret = atomicCAS((atomic64_t*)&comps[idx], comp, 0);
      assert(ret == comp);
      break;
    }
  }

  return comp;
}

__device__ __forceinline__ atomic64_t find_msg_block(volatile atomic64_t* comps, int& ret_idx) {
  __shared__ volatile int idx;
  __shared__ atomic64_t comp;
  if (threadIdx.x == 0) {
    idx = INT_MAX;
    comp = 0;
  }
  __syncthreads();

  // Look for a valid message (traverse once)
  for (int i = threadIdx.x; i < LOCAL_MAX * c_n_sms; i += blockDim.x) {
    if (comps[i] != 0) {
      atomicMin_block((int*)&idx, i);
    }
    __threadfence();
  }
  __syncthreads();
  ret_idx = idx;

  // If a message is found
  if (idx != INT_MAX && threadIdx.x == 0) {
    comp = (atomic64_t)comps[idx];
    __threadfence();

    // Reset message address to zero
    atomic64_t ret = atomicCAS((atomic64_t*)&comps[idx], comp, 0);
    assert(ret == comp);
  }
  __syncthreads();

  return comp;
}

__device__ void charm::comm::process_local() {
  int dst_pe_local = blockIdx.x;

  // Look for valid message addresses
  volatile atomic64_t* dst_recv_comp = recv_comp + LOCAL_MAX * c_n_sms * dst_pe_local;
  int msg_idx;
  composite_t comp((uint64_t)find_msg_block(dst_recv_comp, msg_idx));

  if (comp.data) {
    ringbuf_t* my_mbuf = mbuf + dst_pe_local;
    envelope* env = (envelope*)my_mbuf->addr(comp.offset());
    if (threadIdx.x == 0) {
      PDEBUG("PE %d (SM %d) receiving local message (env %p, msgtype %d, size %llu) "
          "from PE %d (SM %d) at index %d\n",
          (int)s_mem[s_idx::my_pe], dst_pe_local, env, env->type, env->size,
          env->src_pe, msg_idx / LOCAL_MAX, msg_idx % LOCAL_MAX);
    }
    __syncthreads();

    // Process message in parallel
    msgtype type = process_msg(env, nullptr, begin_term_flag, do_term_flag);

    // Signal sender for cleanup
    if (threadIdx.x == 0 && type != msgtype::user) {
      PDEBUG("PE %d (SM %d) signaling local cleanup (env %p, msgtype %d, size %llu) "
          "to PE %d (SM %d) at index %d\n",
          (int)s_mem[s_idx::my_pe], dst_pe_local, env, env->type, env->size,
          env->src_pe, msg_idx / LOCAL_MAX, msg_idx % LOCAL_MAX);

      int src_pe_local = msg_idx / LOCAL_MAX;
      int clup_idx = msg_idx % LOCAL_MAX;
      volatile int* src_send_status = send_status + LOCAL_MAX * c_n_sms * src_pe_local
        + LOCAL_MAX * dst_pe_local;
      int ret = atomicCAS((int*)&src_send_status[clup_idx], SIGNAL_USED, SIGNAL_CLUP);
      assert(ret == SIGNAL_USED);
    }
    __syncthreads();
  }

  /*
  composite_t src_composite;
  void* addr = nullptr;
  size_t src_offset;
  size_t msg_size;

  // Check local composite queue
  compbuf_t* my_recv_local_comp = recv_local_comp + blockIdx.x;
  ringbuf_t* my_mbuf = mbuf + blockIdx.x;
  if (threadIdx.x == 0) {
    if (my_recv_local_comp->count > 0) {
      uint64_t data = *(my_recv_local_comp->addr(my_recv_local_comp->read));
      src_composite = composite_t(data);
      src_offset = src_composite.offset();
      msg_size = src_composite.size();
      addr = my_mbuf->addr(src_offset);
      s_mem[s_idx::src] = (uint64_t)addr; // Store in shared memory for other threads
      PDEBUG("PE %d receiving local message: offset %llu, size %llu\n",
          (int)s_mem[s_idx::my_pe], src_offset, msg_size);
    } else {
      s_mem[s_idx::src] = (uint64_t)nullptr;
    }
  }
  __syncthreads();
  addr = (void*)s_mem[s_idx::src]; // Fetch message address from shared memory

  if (addr) { // Check if we have a message
    // Process message in parallel
    msgtype type = process_msg(addr, nullptr, begin_term_flag, do_term_flag);

    // Message cleanup
    if (threadIdx.x == 0) {
      // Release composite from local queue
      my_recv_local_comp->release();

#ifndef NO_CLEANUP
      if (type != msgtype::user) {
        // Store composite to be cleared from memory
        addr_heap.push(src_composite);
        PDEBUG("PE %d flagging local message for cleanup: offset %llu, "
            "size %llu\n", (int)s_mem[s_idx::my_pe], src_offset, msg_size);
      }
#endif // !NO_CLEANUP
    }
    __syncthreads();
  }
  */
}

__device__ void charm::comm::process_remote() {
  /*
  size_t count = 0;
  void* dst_addr = nullptr;
  int my_pe = s_mem[s_idx::my_pe];

  // Check if there are any message requests
  uint64_t* my_recv_remote_comp = recv_remote_comp + (REMOTE_MSG_COUNT_MAX * c_n_pes) * blockIdx.x;
  size_t* my_recv_remote_comp_idx = recv_remote_comp_idx + (REMOTE_MSG_COUNT_MAX * c_n_pes) * blockIdx.x;
#ifdef NVSHMEM_BLOCK_IMPL
  count = nvshmem_uint64_test_some_block(my_recv_remote_comp, REMOTE_MSG_COUNT_MAX * c_n_pes,
      my_recv_remote_comp_idx, NVSHMEM_CMP_GT, 0);
#endif
  if (threadIdx.x == 0) {
#ifndef NVSHMEM_BLOCK_IMPL
    count = nvshmem_uint64_test_some(my_recv_remote_comp, REMOTE_MSG_COUNT_MAX * c_n_pes,
        my_recv_remote_comp_idx, nullptr, NVSHMEM_CMP_GT, 0);
#endif
    s_mem[s_idx::size] = (uint64_t)count;
  }
  __syncthreads();
  count = (size_t)s_mem[s_idx::size];

  ringbuf_t* my_mbuf = mbuf + blockIdx.x;
  if (count > 0) {
    for (size_t i = 0; i < count; i++) {
      size_t msg_idx;
      size_t msg_size;
      int src_pe;
      ringbuf_off_t dst_offset;

      if (threadIdx.x == 0) {
        // Obtain information about this message request
        msg_idx = my_recv_remote_comp_idx[i];
        uint64_t data = nvshmem_signal_fetch(my_recv_remote_comp + msg_idx);
        composite_t src_composite(data);
        size_t src_offset = src_composite.offset();
        msg_size = src_composite.size();
        src_pe = msg_idx / REMOTE_MSG_COUNT_MAX;
        msg_idx -= REMOTE_MSG_COUNT_MAX * src_pe;

        // Reserve space for incoming message
        dst_offset = my_mbuf->acquire(msg_size);
        if (dst_offset == -1) {
          PERROR("PE %d: Not enough space in message buffer\n", my_pe);
          assert(false);
        }
        PDEBUG("PE %d acquired message: offset %llu, size %llu\n",
            my_pe, dst_offset, msg_size);

        // Perform a get operation to fetch the message
        // TODO: Make asynchronous
        dst_addr = my_mbuf->addr(dst_offset);
        s_mem[s_idx::dst] = (uint64_t)dst_addr;
        int src_pe_local = src_pe % c_n_sms;
        int src_pe_nvshmem = src_pe / c_n_sms;
        ringbuf_t* src_mbuf = mbuf + src_pe_local;
        nvshmem_char_get((char*)dst_addr, (char*)src_mbuf->addr(src_offset),
            msg_size, src_pe_nvshmem);
        PDEBUG("PE %d receiving message: src offset %llu, dst offset %llu, "
            "size %llu, src PE %d, idx %llu\n", my_pe, src_offset, dst_offset,
            msg_size, src_pe, msg_idx);
      }
      __syncthreads();
      dst_addr = (void*)s_mem[s_idx::dst];

      // Process message in parallel
      msgtype type = process_msg(dst_addr, nullptr, begin_term_flag, do_term_flag);

      if (threadIdx.x == 0) {
        // Clear message request
        // FIXME: Need fence after?
        nvshmemx_signal_op(my_recv_remote_comp + REMOTE_MSG_COUNT_MAX * src_pe + msg_idx,
            SIGNAL_FREE, NVSHMEM_SIGNAL_SET, s_mem[s_idx::my_pe_nvshmem]);

        int src_pe_nvshmem = src_pe / c_n_sms;
        int src_pe_local = src_pe % c_n_sms;
        uint64_t* src_send_status = send_status + (REMOTE_MSG_COUNT_MAX * c_n_pes) * src_pe_local;
#ifndef NO_CLEANUP
        // Store composite to be cleared from memory
        composite_t dst_composite(dst_offset, msg_size);
        addr_heap.push(dst_composite);
        PDEBUG("PE %d flagging received message for cleanup: offset %llu, "
            "size %llu, src PE %d, idx %llu\n", my_pe, dst_composite.offset(),
            dst_composite.size(), src_pe, msg_idx);

        // Notify sender that message is ready for cleanup
        int signal = (type == msgtype::user) ? SIGNAL_FREE : SIGNAL_CLUP;
        nvshmemx_signal_op(src_send_status + REMOTE_MSG_COUNT_MAX * my_pe + msg_idx,
            signal, NVSHMEM_SIGNAL_SET, src_pe_nvshmem);
#else
        // Notify sender that message has been delivered
        nvshmemx_signal_op(src_send_status + REMOTE_MSG_COUNT_MAX * my_pe + msg_idx,
            SIGNAL_FREE, NVSHMEM_SIGNAL_SET, src_pe_nvshmem);
#endif // !NO_CLEANUP
      }
      __syncthreads();
    }

    // Reset indices array for next use
    memset_kernel(my_recv_remote_comp_idx, 0, REMOTE_MSG_COUNT_MAX * c_n_pes * sizeof(size_t));
  }
*/
}

__device__ void charm::comm::cleanup() {
  // Cleanup using min-heap
  int src_pe_local = blockIdx.x;
  volatile int* src_send_status = send_status + LOCAL_MAX * c_n_sms * src_pe_local;
  int clup_idx = find_signal_block(src_send_status, LOCAL_MAX * c_n_sms,
      SIGNAL_CLUP, SIGNAL_FREE, false);

  // Push stored composite to min-heap and clean up
  if (clup_idx != INT_MAX && threadIdx.x == 0) {
    atomic64_t* src_send_comp = send_comp + LOCAL_MAX * c_n_sms * src_pe_local;
    composite_t comp(src_send_comp[clup_idx]);
    addr_heap.push(comp);
    PDEBUG("PE %d (SM %d) flagging message for cleanup: offset %llu, size %llu, "
        "dst SM %d, idx %d\n", (int)s_mem[s_idx::my_pe], src_pe_local,
        comp.offset(), comp.size(), clup_idx / LOCAL_MAX, clup_idx % LOCAL_MAX);

    composite_t top;
    size_t clup_offset;
    size_t clup_size;
    ringbuf_t* my_mbuf = mbuf + blockIdx.x;
    // Clean up as many messages as possible
    while (true) {
      top = addr_heap.top();
      if (top.data == UINT64_MAX) break;

      clup_offset = top.offset();
      clup_size = top.size();
      if ((clup_offset == my_mbuf->start_offset + my_mbuf->read) && clup_size > 0) {
        my_mbuf->release(clup_size);
        addr_heap.pop();
        PDEBUG("PE %d (SM %d) releasing message: offset %llu, size %llu\n",
            (int)s_mem[s_idx::my_pe], src_pe_local, clup_offset, clup_size);
      } else break;
    }
  }
  __syncthreads();

  /*
#ifndef NO_CLEANUP
  size_t count = 0;

  // Check for messages that have been delivered to the destination PE
  uint64_t* my_send_status = send_status + (REMOTE_MSG_COUNT_MAX * c_n_pes) * blockIdx.x;
  uint64_t* my_send_status_idx = send_status_idx + (REMOTE_MSG_COUNT_MAX * c_n_pes) * blockIdx.x;
#ifdef NVSHMEM_BLOCK_IMPL
  count = nvshmem_uint64_test_some_block(my_send_status, REMOTE_MSG_COUNT_MAX * c_n_pes,
      my_send_status_idx, NVSHMEM_CMP_EQ, SIGNAL_CLUP);
#endif
  if (threadIdx.x == 0) {
#ifndef NVSHMEM_BLOCK_IMPL
    count = nvshmem_uint64_test_some(my_send_status, REMOTE_MSG_COUNT_MAX * c_n_pes,
        my_send_status_idx, nullptr, NVSHMEM_CMP_EQ, SIGNAL_CLUP);
#endif
    s_mem[s_idx::size] = (uint64_t)count;
  }
  __syncthreads();
  count = (size_t)s_mem[s_idx::size];

  if (count > 0) {
    if (threadIdx.x == 0) {
      for (size_t i = 0; i < count; i++) {
        size_t msg_idx = my_send_status_idx[i];

        // Push stored composite to heap to be cleared from memory
        uint64_t* my_send_comp = send_comp + (REMOTE_MSG_COUNT_MAX * c_n_pes) * blockIdx.x;
        composite_t src_composite(my_send_comp[msg_idx]);
        addr_heap.push(src_composite);
        PDEBUG("PE %d flagging sent message for cleanup: offset %llu, size %llu, "
            "dst PE %llu, idx %llu\n", (int)s_mem[s_idx::my_pe], src_composite.offset(),
            src_composite.size(), msg_idx / REMOTE_MSG_COUNT_MAX,
            msg_idx % REMOTE_MSG_COUNT_MAX);

        // Reset signal to SIGNAL_FREE
        nvshmemx_signal_op(my_send_status + msg_idx, SIGNAL_FREE,
            NVSHMEM_SIGNAL_SET, s_mem[s_idx::my_pe_nvshmem]);
      }
    }
    __syncthreads();

    // Reset indices array for next use
    memset_kernel(my_send_status_idx, 0, REMOTE_MSG_COUNT_MAX * c_n_pes * sizeof(size_t));
  }

  // Clean up messages
  if (threadIdx.x == 0) {
    while (true) {
      composite_t comp = addr_heap.top();
      if (comp.data == UINT64_MAX) break;

      size_t clup_offset = comp.offset();
      size_t clup_size = comp.size();
      ringbuf_t* my_mbuf = mbuf + blockIdx.x;
      if (clup_offset == my_mbuf->read && clup_size > 0) {
        my_mbuf->release(clup_size);
        addr_heap.pop();
        PDEBUG("PE %d releasing message: offset %llu, size %llu\n",
            (int)s_mem[s_idx::my_pe], clup_offset, clup_size);
      } else break;
    }
  }
  __syncthreads();
#endif // !NO_CLEANUP
*/
}

/*
__device__ void charm::message::alloc(int idx, int ep, size_t size) {
  size_t msg_size = envelope::alloc_size(sizeof(regular_msg) + size);
  env = create_envelope(msgtype::user, msg_size, &offset);
}

__device__ void charm::message::free() {
  // TODO
}
*/

// Single-threaded
__device__ envelope* charm::create_envelope(msgtype type, size_t payload_size,
    size_t& offset, int dst_pe) {
  size_t msg_size = envelope::alloc_size(type, payload_size);

  /*
  envelope* env;

  int dst_pe_nvshmem = get_pe_nvshmem(dst_pe);
  if (dst_pe_nvshmem == s_mem[s_idx::my_pe_nvshmem]) {
    // Destination is on the same NVSHMEM PE
    env = (envelope*) new char[msg_size];
    new (env) envelope(type, msg_size, s_mem[s_idx::my_pe]);
    __threadfence();
  } else {
    // TODO: Destination is on a different NVSHMEM PE
  }

  return env;
  */

  // Reserve space for this message in message buffer
  int my_pe = s_mem[s_idx::my_pe];
  ringbuf_t* my_mbuf = mbuf + blockIdx.x;
  bool success = my_mbuf->acquire(msg_size, offset);
  if (!success) {
    PERROR("PE %d: Not enough space in message buffer\n", my_pe);
    assert(false);
  }
  PDEBUG("PE %d acquired message: offset %llu, size %llu\n", my_pe, mbuf_off, msg_size);

  // Create envelope
  return new (my_mbuf->addr(offset)) envelope(type, msg_size, my_pe);
}


__device__ void charm::send_msg(envelope* env, size_t offset, int dst_pe) {
  int src_pe_local = blockIdx.x;
  int dst_pe_local = get_pe_local(dst_pe);
  if (get_pe_nvshmem(dst_pe) == s_mem[s_idx::my_pe_nvshmem]) {
    // Message local to this NVSHMEM PE
    volatile int* src_send_status = send_status + LOCAL_MAX * c_n_sms * src_pe_local
      + LOCAL_MAX * dst_pe_local;

    // Find and reserve free message index
    int free_idx = find_signal_block(src_send_status, LOCAL_MAX, SIGNAL_FREE,
        SIGNAL_USED, true);

    if (threadIdx.x == 0) {
      PDEBUG("PE %d (SM %d) sending local message (env %p, msgtype %d, size %llu) "
          "to PE %d (SM %d) at index %d\n",
          (int)s_mem[s_idx::my_pe], src_pe_local, env, env->type, env->size,
          dst_pe, dst_pe_local, free_idx);

      // Atomically store composite in receiver
      volatile atomic64_t* dst_recv_comp = recv_comp + LOCAL_MAX * c_n_sms * dst_pe_local
        + LOCAL_MAX * src_pe_local;
      composite_t comp(offset, env->size);
      atomic64_t ret = atomicCAS((atomic64_t*)&dst_recv_comp[free_idx], 0, (atomic64_t)comp.data);
      assert(ret == 0);

      // Store composite for later cleanup
      atomic64_t* src_send_comp = send_comp + LOCAL_MAX * c_n_sms * src_pe_local
        + LOCAL_MAX * dst_pe_local;
      src_send_comp[free_idx] = (atomic64_t)comp.data;
    }
    __syncthreads();
  } else {
    // TODO
  }
  /*
  // Create composite using offset and size of source buffer
  int my_pe = s_mem[s_idx::my_pe];

  if (dst_pe == my_pe) {
    // Sending message to itself
    // Acquire space in local composite queue and store composite
    compbuf_t* my_recv_local_comp = recv_local_comp + blockIdx.x;
    compbuf_off_t local_offset = my_recv_local_comp->acquire();
    if (local_offset < 0) {
      PERROR("Out of space in local composite queue\n");
      assert(false);
    }
    *(composite_t*)my_recv_local_comp->addr(local_offset) = composite_t(offset, env->size);
    PDEBUG("PE %d sending local message: offset %llu, local %lld, size %llu\n",
        my_pe, offset, local_offset, env->size);
  } else {
    // Sending message to a different PE
    // Obtain a message index for the target PE and set the corresponding used signal
    uint64_t* my_send_status = send_status + (REMOTE_MSG_COUNT_MAX * c_n_pes) * blockIdx.x;
    size_t send_offset = REMOTE_MSG_COUNT_MAX * dst_pe;
    my_send_status += send_offset;
    size_t msg_idx = nvshmem_uint64_wait_until_any(my_send_status, REMOTE_MSG_COUNT_MAX,
        nullptr, NVSHMEM_CMP_EQ, SIGNAL_FREE);
    // Reserve send status
    nvshmemx_signal_op(my_send_status + msg_idx, SIGNAL_USED,
        NVSHMEM_SIGNAL_SET, s_mem[s_idx::my_pe_nvshmem]);

    // Send composite
    int dst_pe_local = dst_pe % c_n_sms;
    uint64_t* dst_recv_remote_comp = recv_remote_comp + (REMOTE_MSG_COUNT_MAX * c_n_pes) * dst_pe_local;
    size_t recv_offset = REMOTE_MSG_COUNT_MAX * my_pe;
    dst_recv_remote_comp += recv_offset;
    int dst_pe_nvshmem = dst_pe / c_n_sms;
    composite_t src_composite(offset, env->size);
    nvshmemx_signal_op(dst_recv_remote_comp + msg_idx, src_composite.data,
        NVSHMEM_SIGNAL_SET, dst_pe_nvshmem);
    assert(msg_idx != SIZE_MAX);
    PDEBUG("PE %d sending message request: offset %llu, size %llu, dst PE %d, idx %llu\n",
        my_pe, offset, env->size, dst_pe, msg_idx);

#ifndef NO_CLEANUP
    // Store source composite for later cleanup
    uint64_t* my_send_comp = send_comp + (REMOTE_MSG_COUNT_MAX * c_n_pes) * blockIdx.x;
    my_send_comp[send_offset + msg_idx] = src_composite.data;
#endif
  }
  */
}

__device__ void charm::send_reg_msg(int chare_id, int chare_idx, int ep_id,
    void* buf, size_t payload_size, int dst_pe) {
  //cudatp_t start = cuda::std::chrono::system_clock::now();

  if (threadIdx.x == 0) {
    envelope* env = create_envelope(msgtype::regular, payload_size,
        (size_t&)s_mem[s_idx::offset], dst_pe);
    s_mem[s_idx::env] = (uint64_t)env;

    regular_msg* msg = new ((char*)env + sizeof(envelope)) regular_msg(chare_id,
      chare_idx, ep_id);

    if (payload_size > 0) {
      s_mem[s_idx::dst] = (uint64_t)((char*)msg + sizeof(regular_msg));
      s_mem[s_idx::src] = (uint64_t)buf;
      s_mem[s_idx::size] = (uint64_t)payload_size;
    }
  }
  __syncthreads();

  /*
  cudatp_t mid1 = cuda::std::chrono::system_clock::now();
  cudatp_dur_t dur1 = mid1 - start;
  */

  // Fill in payload (from regular GPU memory to NVSHMEM symmetric memory)
  if (payload_size > 0) {
    memcpy_kernel((void*)s_mem[s_idx::dst], (void*)s_mem[s_idx::src],
        (size_t)s_mem[s_idx::size]);
  }

  /*
  cudatp_t mid2 = cuda::std::chrono::system_clock::now();
  cudatp_dur_t dur2 = mid2 - mid1;
  */

  send_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset], dst_pe);

  /*
  cudatp_t end = cuda::std::chrono::system_clock::now();
  cudatp_dur_t dur3 = end - mid2;
  cudatp_dur_t dur_total = end - start;
  if (threadIdx.x == 0) {
    printf("!!! send_reg_msg %.3lf us (create_envelope %.3lf us, memcpy_kernel %.3lf us, "
        "send_msg %.3lf us)\n",
        dur_total.count() * 1e6, dur1.count() * 1e6, dur2.count() * 1e6, dur3.count() * 1e6);
  }
  */
}

__device__ __forceinline__ void send_user_msg_common(int chare_id, int chare_idx,
    int ep_id, const message& msg) {
  envelope* env = msg.env;
  if (threadIdx.x == 0) {
    // Set regular message fields using placement new
    new ((char*)env + sizeof(envelope)) regular_msg(chare_id, chare_idx, ep_id);
  }
  __syncthreads();

  send_msg(env, msg.offset, msg.dst_pe);
}

__device__ void charm::send_user_msg(int chare_id, int chare_idx, int ep_id,
    const message& msg) {
  send_user_msg_common(chare_id, chare_idx, ep_id, msg);
}

__device__ void charm::send_user_msg(int chare_id, int chare_idx, int ep_id,
    const message& msg, size_t payload_size) {
  // Send size can be smaller than allocated message size
  msg.env->size = envelope::alloc_size(msgtype::user, payload_size);

  send_user_msg_common(chare_id, chare_idx, ep_id, msg);
}

__device__ void charm::send_term_msg(bool begin, int dst_pe) {
  if (threadIdx.x == 0) {
    envelope* env = create_envelope(
        begin ? msgtype::begin_terminate : msgtype::do_terminate, 0,
        (size_t&)s_mem[s_idx::offset], dst_pe);
    s_mem[s_idx::env] = (uint64_t)env;
  }
  __syncthreads();

  send_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset], dst_pe);
}
