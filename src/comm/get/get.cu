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
#define NVSHMEM_BLOCK_EXT

// Use thread block extension of NVSHMEM for communication
// Turned off for now as it shows identical performance
//#define NVSHMEM_BLOCK_COMM

#define MBUF_PE_SIZE 8388608 // 8MB per PE
#define MBUF_CE_SIZE 134217728 // 128MB per CE

/*
// Maximum number of messages that are allowed to be in flight per pair of PEs
#define REMOTE_MSG_COUNT_MAX 4
// Maximum number of messages that can be stored in the local message queue
#define LOCAL_MSG_COUNT_MAX 128
*/
#define LOCAL_MSG_MAX 4
#define REMOTE_MSG_MAX 4

using namespace charm;

typedef unsigned long long int atomic64_t;

extern cudaStream_t stream;

// Managed memory (actual data may reside elsewhere)
__managed__ void* nvshmem_buf; // NVSHMEM

__managed__ ringbuf_t* mbuf; // Managed

__managed__ volatile int* send_status_local; // Global
__managed__ uint64_t* send_status_remote; // NVSHMEM
__managed__ size_t* send_status_remote_idx; // Global

__managed__ uint64_t* send_comp_local; // Global
__managed__ uint64_t* send_comp_remote; // Global

__managed__ volatile atomic64_t* recv_comp_local; // Global
__managed__ uint64_t* recv_comp_remote; // NVSHMEM
__managed__ size_t* recv_comp_remote_idx; // Global

__managed__ composite_t* heap_buf; // Global

// GPU shared memory
extern __shared__ uint64_t s_mem[];

enum {
  SIGNAL_FREE = 0,
  SIGNAL_USED = 1,
  SIGNAL_CLUP = 2
};

void charm::comm_init_host(int n_sms, int n_pes, int n_ces, int n_clusters_dev,
    int n_pes_cluster, int n_ces_cluster) {
  // Allocate NVSHMEM message buffer
  size_t mbuf_cluster_size = MBUF_PE_SIZE * n_pes_cluster + MBUF_CE_SIZE * n_ces_cluster;
  nvshmem_buf = nvshmem_malloc(mbuf_cluster_size * n_clusters_dev);
  assert(nvshmem_buf);
  cudaMallocManaged(&mbuf, sizeof(ringbuf_t) * n_sms);
  assert(mbuf);
  ringbuf_t* cur_mbuf = mbuf;
  size_t start_offset = 0;
  for (int i = 0; i < n_sms; i++) {
    int cluster_size = n_pes_cluster + n_ces_cluster;
    int rank_in_cluster = i % cluster_size;
    bool is_pe = rank_in_cluster < n_pes_cluster;
    size_t mbuf_size = is_pe ? MBUF_PE_SIZE : MBUF_CE_SIZE;
    cur_mbuf->init(nvshmem_buf, start_offset, mbuf_size);
    start_offset += mbuf_size;
    cur_mbuf++;
  }

  // Allocate data structures
  int n_ces_dev = n_ces_cluster * n_clusters_dev;
  size_t local_count = LOCAL_MSG_MAX * n_sms * n_sms;
  size_t remote_count = REMOTE_MSG_MAX * n_ces * n_ces_dev;
  size_t status_local_size = sizeof(int) * local_count;
  size_t status_remote_size = sizeof(uint64_t) * remote_count;
  size_t idx_size = sizeof(size_t) * remote_count;
  size_t comp_local_size = sizeof(atomic64_t) * local_count;
  size_t comp_remote_size = sizeof(atomic64_t) * remote_count;
  size_t heap_size = sizeof(composite_t) * local_count * 2;
  assert(sizeof(atomic64_t) == sizeof(uint64_t));
  cudaMalloc(&send_status_local, status_local_size);
  send_status_remote = (uint64_t*)nvshmem_malloc(status_remote_size);
  cudaMalloc(&send_status_remote_idx, idx_size);
  cudaMalloc(&send_comp_local, comp_local_size);
  cudaMalloc(&send_comp_remote, comp_remote_size);
  cudaMalloc(&recv_comp_local, comp_local_size);
  recv_comp_remote = (uint64_t*)nvshmem_malloc(comp_remote_size);
  cudaMalloc(&recv_comp_remote_idx, idx_size);
  cudaMalloc(&heap_buf, heap_size);
  assert(send_status_local && send_status_remote && send_status_remote_idx
      && send_comp_local && send_comp_remote && recv_comp_local
      && recv_comp_remote && recv_comp_remote_idx && heap_buf);

  // Clear data structures
  cudaMemsetAsync((void*)send_status_local, 0, status_local_size, stream);
  cudaMemsetAsync((void*)send_status_remote, 0, status_remote_size, stream);
  cudaMemsetAsync((void*)send_status_remote_idx, 0, idx_size, stream);
  cudaMemsetAsync((void*)send_comp_local, 0, comp_local_size, stream);
  cudaMemsetAsync((void*)send_comp_remote, 0, comp_remote_size, stream);
  cudaMemsetAsync((void*)recv_comp_local, 0, comp_local_size, stream);
  cudaMemsetAsync((void*)recv_comp_remote, 0, comp_remote_size, stream);
  cudaMemsetAsync((void*)recv_comp_remote_idx, 0, idx_size, stream);
  cudaMemsetAsync((void*)heap_buf, 0, heap_size, stream);
  cudaStreamSynchronize(stream);
  cuda_check_error();
}

void charm::comm_fini_host() {
  // Free NVSHMEM message buffer
  nvshmem_free(nvshmem_buf);

  // Free data structures
  cudaFree((void*)mbuf);
  cudaFree((void*)send_status_local);
  nvshmem_free(send_status_remote);
  cudaFree((void*)send_status_remote_idx);
  cudaFree((void*)send_comp_local);
  cudaFree((void*)send_comp_remote);
  cudaFree((void*)recv_comp_local);
  nvshmem_free(recv_comp_remote);
  cudaFree((void*)recv_comp_remote_idx);
  cudaFree((void*)heap_buf);
}

// Single-threaded
__device__ void charm::comm::init() {
  // Initialize min-heap
  int comp_count = LOCAL_MSG_MAX * c_n_sms * 2;
  composite_t* my_heap_buf = heap_buf + comp_count * blockIdx.x;
  addr_heap.init(my_heap_buf, comp_count);

  sent_term_flag = false;
  begin_term_flag = false;
  do_term_flag = false;

  if (!s_mem[s_idx::is_pe]) {
    // Store local ranks and count of child PEs for this CE
    child_count = 0;
    child_local_ranks = new int[c_n_pes_cluster];
    assert(child_local_ranks);

    int my_cluster = blockIdx.x / c_cluster_size;
    int my_rank_in_cluster = blockIdx.x % c_cluster_size;
    int start_local_rank = my_cluster * c_cluster_size;
    for (int i = 0; i < c_n_pes_cluster; i++) {
      int ce_rank_in_cluster = i % c_n_ces_cluster + c_n_pes_cluster;
      if (ce_rank_in_cluster == my_rank_in_cluster) {
        child_local_ranks[child_count++] = start_local_rank + i;
      }
    }
  }
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
  for (int i = 0; i < LOCAL_MSG_MAX * c_n_sms; i++) {
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
  for (int i = threadIdx.x; i < LOCAL_MSG_MAX * c_n_sms; i += blockDim.x) {
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
  int dst_local_rank = blockIdx.x;
  bool is_pe = (s_mem[s_idx::is_pe] == 1);
#ifdef DEBUG
  int dst_elem = is_pe ? s_mem[s_idx::my_pe] : s_mem[s_idx::my_ce];
#endif

  // Look for valid message addresses
  volatile atomic64_t* recv_comp = recv_comp_local + LOCAL_MSG_MAX * c_n_sms * dst_local_rank;
  int msg_idx;
  composite_t comp((uint64_t)find_msg_block(recv_comp, msg_idx));

  if (comp.data) {
    int src_local_rank = msg_idx / LOCAL_MSG_MAX;
    int clup_idx = msg_idx % LOCAL_MSG_MAX;
    ringbuf_t* dst_mbuf = mbuf + dst_local_rank;
    envelope* env = (envelope*)dst_mbuf->addr(comp.offset());
    if (threadIdx.x == 0) {
      PDEBUG("%s %d receiving local message (env %p, msgtype %d, size %llu) "
          "from local rank %d at index %d\n",
          is_pe ? "PE" : "CE", dst_elem, env, env->type, env->size,
          src_local_rank, clup_idx);
    }
    __syncthreads();

    // Process message in parallel
    msgtype type = is_pe ?
      process_msg_pe(env, comp.offset(), begin_term_flag, do_term_flag)
      : process_msg_ce(env, comp.offset(), sent_term_flag, begin_term_flag,
          do_term_flag);

    // Signal sender for cleanup
    if (threadIdx.x == 0 && type != msgtype::user) {
      PDEBUG("%s %d process_local signal cleanup (env %p, msgtype %d, size %llu) "
          "to local rank %d at index %d\n",
          is_pe ? "PE" : "CE", dst_elem, env, env->type, env->size,
          src_local_rank, clup_idx);

      volatile int* src_send_status = send_status_local
        + LOCAL_MSG_MAX * c_n_sms * src_local_rank + LOCAL_MSG_MAX * dst_local_rank;
#ifndef NO_CLEANUP
      int signal = SIGNAL_CLUP;
#else
      int signal = SIGNAL_FREE;
#endif
      int ret = atomicCAS((int*)&src_send_status[clup_idx], SIGNAL_USED, signal);
      assert(ret == SIGNAL_USED);
    }
    __syncthreads();
  }
}

__device__ void charm::comm::process_remote() {
  int dst_local_rank = blockIdx.x;
  int dst_ce = s_mem[s_idx::my_ce];
  int dst_ce_dev = get_ce_in_dev(dst_ce);
  int dst_dev = get_dev_from_ce(dst_ce);

  // Check if there are any incoming messages
  uint64_t* recv_comp = recv_comp_remote + (REMOTE_MSG_MAX * c_n_ces) * dst_ce_dev;
  size_t* recv_comp_idx = recv_comp_remote_idx + (REMOTE_MSG_MAX * c_n_ces) * dst_ce_dev;
  size_t count = 0;
#ifdef NVSHMEM_BLOCK_EXT
  count = nvshmem_uint64_test_some_block(recv_comp, REMOTE_MSG_MAX * c_n_ces,
      recv_comp_idx, NVSHMEM_CMP_GT, 0);
#else
  if (threadIdx.x == 0) {
    count = nvshmem_uint64_test_some(recv_comp, REMOTE_MSG_MAX * c_n_ces,
        recv_comp_idx, nullptr, NVSHMEM_CMP_GT, 0);
    s_mem[s_idx::size] = (uint64_t)count;
  }
  __syncthreads();
  count = (size_t)s_mem[s_idx::size];
#endif

  if (count > 0) {
    ringbuf_t* dst_mbuf = mbuf + dst_local_rank;;
    size_t found_idx;
    uint64_t data;
    size_t src_offset;
    size_t msg_size;
    int src_ce;
    int src_ce_dev;
    int src_dev;
    size_t msg_idx;
    size_t dst_offset;
    void* dst_addr = nullptr;
    void* src_addr = nullptr;

    for (size_t i = 0; i < count; i++) {
      if (threadIdx.x == 0) {
        // Obtain information about this message
        found_idx = recv_comp_idx[i];
        data = nvshmem_signal_fetch(recv_comp + found_idx);
        composite_t src_composite(data);
        src_offset = src_composite.offset();
        msg_size = src_composite.size();
        src_ce = found_idx / REMOTE_MSG_MAX;
        msg_idx = found_idx % REMOTE_MSG_MAX;

        // Reserve space for incoming message
        bool success = dst_mbuf->acquire(msg_size, dst_offset);
        if (!success) {
          PERROR("CE %d: Not enough space in message buffer\n", dst_ce);
          assert(false);
        }
        PDEBUG("CE %d acquired space for incoming remote message: offset %llu, size %llu\n",
            dst_ce, dst_offset, msg_size);

        // Perform a get operation to fetch the message
        // TODO: Make asynchronous
        dst_addr = dst_mbuf->addr(dst_offset);
        src_addr = dst_mbuf->addr(src_offset);
        src_ce_dev = get_ce_in_dev(src_ce);
        src_dev = get_dev_from_ce(src_ce);
        s_mem[s_idx::dst] = (uint64_t)dst_addr;
        s_mem[s_idx::offset] = (uint64_t)dst_offset;
#ifdef NVSHMEM_BLOCK_COMM
        s_mem[s_idx::src] = (uint64_t)src_addr;
        s_mem[s_idx::size] = (uint64_t)msg_size;
        s_mem[s_idx::dev] = (uint64_t)src_dev;
#else
        nvshmem_char_get((char*)dst_addr, (char*)src_addr, msg_size, src_dev);
#endif
        PDEBUG("CE %d remote get: src offset %llu, dst offset %llu, "
            "size %llu, src CE %d, idx %llu\n", dst_ce, src_offset, dst_offset,
            msg_size, src_ce, msg_idx);
      }
      __syncthreads();
      dst_addr = (void*)s_mem[s_idx::dst];
      dst_offset = (size_t)s_mem[s_idx::offset];
#ifdef NVSHMEM_BLOCK_COMM
      src_addr = (void*)s_mem[s_idx::src];
      msg_size = (size_t)s_mem[s_idx::size];
      src_dev = (int)s_mem[s_idx::dev];
      nvshmemx_char_get_block((char*)dst_addr, (char*)src_addr, msg_size, src_dev);
#endif

      // Process message in parallel
      msgtype type = process_msg_ce(dst_addr, dst_offset, sent_term_flag,
          begin_term_flag, do_term_flag);

      if (threadIdx.x == 0) {
        // Clear message request
        // FIXME: Need fence after?
        nvshmemx_signal_op(recv_comp + found_idx, SIGNAL_FREE, NVSHMEM_SIGNAL_SET,
            dst_dev);

        uint64_t* src_send_status = send_status_remote
          + (REMOTE_MSG_MAX * c_n_ces) * src_ce_dev
          + REMOTE_MSG_MAX * dst_ce;
        int signal = SIGNAL_FREE;
#ifndef NO_CLEANUP
        // Store composite to be cleared from memory
        composite_t dst_composite(dst_offset, msg_size);
        // Forwarded message should not be freed here
        // It will be freed as a local message once it arrives on the destination PE
        if (type != msgtype::forward) {
          addr_heap.push(dst_composite);
        }
        signal = (type == msgtype::user) ? SIGNAL_FREE : SIGNAL_CLUP;
        PDEBUG("CE %d process_remote signal cleanup & push to heap: signal %d, "
            "offset %llu, size %llu, src CE %d, idx %llu\n", dst_ce, signal,
            dst_composite.offset(), dst_composite.size(), src_ce, msg_idx);
#endif
        // Notify sender that message has been delivered
        nvshmemx_signal_op(src_send_status + msg_idx, signal, NVSHMEM_SIGNAL_SET,
            src_dev);
      }
      __syncthreads();
    }

    // Reset indices array for next use
    memset_kernel(recv_comp_idx, 0, REMOTE_MSG_MAX * c_n_ces * sizeof(size_t));
  }
}

__device__ void charm::comm::cleanup_local() {
  int local_rank = blockIdx.x;
  volatile int* send_status = send_status_local + LOCAL_MSG_MAX * c_n_sms * local_rank;
  int clup_idx = find_signal_block(send_status, LOCAL_MSG_MAX * c_n_sms,
      SIGNAL_CLUP, SIGNAL_FREE, false);

  // If a message needs to be cleaned up, add composite to min-heap
  if (clup_idx != INT_MAX && threadIdx.x == 0) {
    uint64_t* send_comp = send_comp_local + LOCAL_MSG_MAX * c_n_sms * local_rank;
    composite_t comp(send_comp[clup_idx]);
    addr_heap.push(comp);
    PDEBUG("%s %d cleanup_local push to heap: "
        "offset %llu, size %llu, dst local rank %d, msg idx %d\n",
        s_mem[s_idx::is_pe] ? "PE" : "CE",
        s_mem[s_idx::is_pe] ? (int)s_mem[s_idx::my_pe] : (int)s_mem[s_idx::my_ce],
        comp.offset(), comp.size(), clup_idx / LOCAL_MSG_MAX,
        clup_idx % LOCAL_MSG_MAX);
  }
  __syncthreads();
}

__device__ void charm::comm::cleanup_remote() {
  int my_ce_dev = get_ce_in_dev(s_mem[s_idx::my_ce]);
  int my_dev = get_dev_from_ce(s_mem[s_idx::my_ce]);

  // Check for messages that have been delivered to the destination PE
  uint64_t* send_status = send_status_remote + (REMOTE_MSG_MAX * c_n_ces) * my_ce_dev;
  uint64_t* send_status_idx = send_status_remote_idx + (REMOTE_MSG_MAX * c_n_ces) * my_ce_dev;
  size_t count = 0;
#ifdef NVSHMEM_BLOCK_EXT
  count = nvshmem_uint64_test_some_block(send_status, REMOTE_MSG_MAX * c_n_ces,
      send_status_idx, NVSHMEM_CMP_EQ, SIGNAL_CLUP);
#else
  if (threadIdx.x == 0) {
    count = nvshmem_uint64_test_some(send_status, REMOTE_MSG_MAX * c_n_ces,
        send_status_idx, nullptr, NVSHMEM_CMP_EQ, SIGNAL_CLUP);
    s_mem[s_idx::size] = (uint64_t)count;
  }
  __syncthreads();
  count = (size_t)s_mem[s_idx::size];
#endif

  // Push composites to min-heap for cleanup
  if (count > 0) {
    if (threadIdx.x == 0) {
      uint64_t* send_comp = send_comp_remote + (REMOTE_MSG_MAX * c_n_ces) * my_ce_dev;

      for (size_t i = 0; i < count; i++) {
        size_t found_idx = send_status_idx[i];
        composite_t src_composite(send_comp[found_idx]);
        addr_heap.push(src_composite);
        PDEBUG("CE %d cleanup_remote push to heap: offset %llu, size %llu, "
            "dst CE %llu, idx %llu, found_idx %llu\n", (int)s_mem[s_idx::my_ce],
            src_composite.offset(), src_composite.size(),
            found_idx / REMOTE_MSG_MAX, found_idx % REMOTE_MSG_MAX, found_idx);

        // Reset signal to SIGNAL_FREE
        nvshmemx_signal_op(send_status + found_idx, SIGNAL_FREE,
            NVSHMEM_SIGNAL_SET, my_dev);
      }
    }
    __syncthreads();

    // Reset indices array for next use
    memset_kernel(send_status_idx, 0, REMOTE_MSG_MAX * c_n_ces * sizeof(size_t));
  }
}

__device__ void charm::comm::cleanup_heap() {
  // Check min-heap and free messages
  if (threadIdx.x == 0) {
    int local_rank = blockIdx.x;
    composite_t top;
    size_t clup_offset;
    size_t clup_size;
    ringbuf_t* my_mbuf = mbuf + local_rank;
    while (true) {
      top = addr_heap.top();
      if (top.data == UINT64_MAX) break;

      clup_offset = top.offset();
      clup_size = top.size();
      if ((clup_offset == my_mbuf->start_offset + my_mbuf->read) && clup_size > 0) {
        bool success = my_mbuf->release(clup_size);
        if (!success) {
          PERROR("%s %d failed to release message: offset %llu, size %llu\n",
              s_mem[s_idx::is_pe] ? "PE" : "CE",
              s_mem[s_idx::is_pe] ? (int)s_mem[s_idx::my_pe] : (int)s_mem[s_idx::my_ce],
              clup_offset, clup_size);
          my_mbuf->print();
          assert(false);
        }
        addr_heap.pop();
        PDEBUG("%s %d releasing message: offset %llu, size %llu\n",
            s_mem[s_idx::is_pe] ? "PE" : "CE",
            s_mem[s_idx::is_pe] ? (int)s_mem[s_idx::my_pe] : (int)s_mem[s_idx::my_ce],
            clup_offset, clup_size);
      } else break;
    }
  }
  __syncthreads();
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
    size_t& offset) {
  size_t msg_size = envelope::alloc_size(type, payload_size);

  // Reserve space for this message in message buffer
  bool is_pe = (s_mem[s_idx::is_pe] == 1);
  int my_elem = is_pe ? s_mem[s_idx::my_pe] : s_mem[s_idx::my_ce];
  ringbuf_t* src_mbuf = mbuf + blockIdx.x;
  bool success = src_mbuf->acquire(msg_size, offset);
  if (!success) {
    PERROR("%s %d: Not enough space in message buffer\n",
        is_pe ? "PE" : "CE", my_elem);
    src_mbuf->print();
    assert(false);
  }
  PDEBUG("%s %d acquired message: offset %llu, size %llu\n",
      is_pe ? "PE" : "CE", my_elem, offset, msg_size);

  // Create envelope
  return new (src_mbuf->addr(offset)) envelope(type, msg_size);
}

__device__ void charm::send_local_msg(envelope* env, size_t offset, int dst_local_rank) {
  int src_local_rank = blockIdx.x;
  volatile int* send_status = send_status_local
    + LOCAL_MSG_MAX * c_n_sms * src_local_rank + LOCAL_MSG_MAX * dst_local_rank;

  // Find and reserve free message index
  int free_idx = find_signal_block(send_status, LOCAL_MSG_MAX, SIGNAL_FREE,
      SIGNAL_USED, true);

  if (threadIdx.x == 0) {
    PDEBUG("%s %d sending local message: dst local rank %d, "
        "index %d, env %p, msgtype %d, size %llu\n", s_mem[s_idx::is_pe] ? "PE" : "CE",
        s_mem[s_idx::is_pe] ? (int)s_mem[s_idx::my_pe] : (int)s_mem[s_idx::my_ce],
        dst_local_rank, free_idx, env, env->type, env->size);

    // Atomically store composite in receiver
    volatile atomic64_t* recv_comp = recv_comp_local
      + LOCAL_MSG_MAX * c_n_sms * dst_local_rank + LOCAL_MSG_MAX * src_local_rank;
    composite_t comp(offset, env->size);
    atomic64_t ret = atomicCAS((atomic64_t*)&recv_comp[free_idx], 0,
        (atomic64_t)comp.data);
    assert(ret == 0);

#ifndef NO_CLEANUP
    // Store composite for later cleanup
    uint64_t* send_comp = send_comp_local
      + LOCAL_MSG_MAX * c_n_sms * src_local_rank + LOCAL_MSG_MAX * dst_local_rank;
    send_comp[free_idx] = comp.data;
#endif
  }
  __syncthreads();
}

__device__ void charm::send_remote_msg(envelope* env, size_t offset, int dst_ce) {
  if (threadIdx.x == 0) {
    int src_ce = s_mem[s_idx::my_ce];
    int src_ce_dev = get_ce_in_dev(src_ce);
    int dst_ce_dev = get_ce_in_dev(dst_ce);
    int dst_dev = get_dev_from_ce(dst_ce);

    // Obtain a message index for the target CE and set signal to used
    uint64_t* send_status = send_status_remote + (REMOTE_MSG_MAX * c_n_ces) * src_ce_dev
      + REMOTE_MSG_MAX * dst_ce;
    size_t msg_idx = nvshmem_uint64_wait_until_any(send_status, REMOTE_MSG_MAX,
        nullptr, NVSHMEM_CMP_EQ, SIGNAL_FREE);
    nvshmemx_signal_op(send_status + msg_idx, SIGNAL_USED, NVSHMEM_SIGNAL_SET, c_my_dev);

    // Send composite
    uint64_t* recv_comp = recv_comp_remote + (REMOTE_MSG_MAX * c_n_ces) * dst_ce_dev
      + REMOTE_MSG_MAX * src_ce;
    composite_t src_composite(offset, env->size);
    nvshmemx_signal_op(recv_comp + msg_idx, src_composite.data, NVSHMEM_SIGNAL_SET,
        dst_dev);
    PDEBUG("CE %d sending remote message: offset %llu, size %llu, dst CE %d, idx %llu\n",
        src_ce, offset, env->size, dst_ce, msg_idx);

#ifndef NO_CLEANUP
    // Store source composite for later cleanup
    uint64_t* send_comp = send_comp_remote + (REMOTE_MSG_MAX * c_n_ces) * src_ce_dev
      + REMOTE_MSG_MAX * dst_ce;
    send_comp[msg_idx] = src_composite.data;
#endif
  }
  __syncthreads();
}

__device__ void charm::send_reg_msg(int chare_id, int chare_idx, int ep_id,
    void* buf, size_t payload_size, int dst_pe) {
  if (threadIdx.x == 0) {
    int src_dev = get_dev_from_pe(s_mem[s_idx::my_pe]);
    int dst_dev = get_dev_from_pe(dst_pe);
    envelope* env = nullptr;
    if (src_dev == dst_dev) {
      // Need to send to another PE on same device
      env = create_envelope(msgtype::regular, payload_size,
          (size_t&)s_mem[s_idx::offset]);

      regular_msg* msg = new ((char*)env + sizeof(envelope)) regular_msg(
          chare_id, chare_idx, ep_id);

      if (payload_size > 0) {
        s_mem[s_idx::dst] = (uint64_t)((char*)msg + sizeof(regular_msg));
        s_mem[s_idx::src] = (uint64_t)buf;
      }
      s_mem[s_idx::size] = (uint64_t)payload_size;
      s_mem[s_idx::local_rank] = (uint64_t)get_local_rank_from_pe(dst_pe);
    } else {
      // Need to send to another device
      // Create and send request to CE
      env = create_envelope(msgtype::request, 0, (size_t&)s_mem[s_idx::offset]);

      request_msg* msg = new ((char*)env + sizeof(envelope)) request_msg(
          chare_id, chare_idx, ep_id, msgtype::regular, buf, payload_size, dst_pe);
      s_mem[s_idx::size] = 0;
      int my_ce = get_ce_from_pe((int)s_mem[s_idx::my_pe]);
      s_mem[s_idx::local_rank] = (uint64_t)get_local_rank_from_ce(my_ce);
    }
    s_mem[s_idx::env] = (uint64_t)env;
  }
  __syncthreads();

  // Fill in payload (from regular GPU memory to NVSHMEM symmetric memory)
  if (s_mem[s_idx::size] > 0) {
    memcpy_kernel((void*)s_mem[s_idx::dst], (void*)s_mem[s_idx::src],
        (size_t)s_mem[s_idx::size]);
  }

  // Send a local message either directly to dst PE or to a responsible CE
  send_local_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset],
      (int)s_mem[s_idx::local_rank]);
}

__device__ void charm::send_delegate_msg(request_msg* req) {
  // Prepare message for sending remotely
  if (threadIdx.x == 0) {
    envelope* env = nullptr;
    if (req->type == msgtype::regular) {
      // Need to send to CE responsible for target PE
      env = create_envelope(msgtype::forward, req->payload_size,
          (size_t&)s_mem[s_idx::offset]);

      forward_msg* msg = new ((char*)env + sizeof(envelope)) forward_msg(
          req->chare_id, req->chare_idx, req->ep_id, req->dst_pe);

      if (req->payload_size > 0) {
        s_mem[s_idx::dst] = (uint64_t)((char*)msg + sizeof(forward_msg));
        s_mem[s_idx::src] = (uint64_t)req->buf;
        s_mem[s_idx::size] = (uint64_t)req->payload_size;
      }
      s_mem[s_idx::dst_ce] = (uint64_t)get_ce_from_pe(req->dst_pe);
    } else if (req->type == msgtype::begin_terminate) {
      // Send begin termination message to CE 0
      env = create_envelope(msgtype::begin_terminate, 0,
          (size_t&)s_mem[s_idx::offset]);

      s_mem[s_idx::dst_ce] = 0;
    } else {
      PERROR("CE %d invalid message type %d in send_delegate_msg\n",
          (int)s_mem[s_idx::my_ce], req->type);
      assert(false);
    }
    s_mem[s_idx::env] = (uint64_t)env;
  }
  __syncthreads();

  // Fill in payload
  if (req->payload_size > 0) {
    memcpy_kernel((void*)s_mem[s_idx::dst], (void*)s_mem[s_idx::src],
        (size_t)s_mem[s_idx::size]);
  }

  send_remote_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset],
      (int)s_mem[s_idx::dst_ce]);
}

// TODO
__device__ __forceinline__ void send_user_msg_common(int chare_id, int chare_idx,
    int ep_id, const message& msg) {
  envelope* env = msg.env;
  if (threadIdx.x == 0) {
    // Set regular message fields using placement new
    new ((char*)env + sizeof(envelope)) regular_msg(chare_id, chare_idx, ep_id);
  }
  __syncthreads();

  //send_msg(env, msg.offset, msg.dst_pe);
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

__device__ void charm::send_begin_term_msg() {
  comm* c = (comm*)(s_mem + SMEM_CNT_MAX);

  // Don't do anything if message to begin termination
  // has already been sent from this PE
  if (c->sent_term_flag) return;

  if (threadIdx.x == 0) {
    c->sent_term_flag = true;

    int src_pe = s_mem[s_idx::my_pe];
    int src_dev = get_dev_from_pe(src_pe);
    int dst_dev = get_dev_from_ce(0);
    envelope* env = nullptr;
    if (src_dev == dst_dev) {
      // CE 0 is on the same device, send directly
      env = create_envelope(msgtype::begin_terminate, 0,
          (size_t&)s_mem[s_idx::offset]);
      s_mem[s_idx::local_rank] = (uint64_t)get_local_rank_from_ce(0);
    } else {
      // CE 0 is on a difference device, delegate to CE
      env = create_envelope(msgtype::request, 0,
          (size_t&)s_mem[s_idx::offset]);

      request_msg* msg = new ((char*)env + sizeof(envelope)) request_msg(
          -1, -1, -1, msgtype::begin_terminate, nullptr, 0, -1);
      int src_ce = get_ce_from_pe(src_pe);
      s_mem[s_idx::local_rank] = (uint64_t)get_local_rank_from_ce(src_ce);
    }
    s_mem[s_idx::env] = (uint64_t)env;
  }
  __syncthreads();

  // Send message to CE 0 (either directly or indirectly)
  send_local_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset],
      (int)s_mem[s_idx::local_rank]);
}

__device__ void charm::send_do_term_msg_ce(int dst_ce) {
  // Prepare message
  if (threadIdx.x == 0) {
    envelope* env = create_envelope(msgtype::do_terminate, 0,
        (size_t&)s_mem[s_idx::offset]);
    s_mem[s_idx::env] = (uint64_t)env;
  }
  __syncthreads();

  // Send message (dst CE could be on the same device or remote)
  int src_ce = s_mem[s_idx::my_ce];
  int src_dev = get_dev_from_ce(src_ce);
  int dst_dev = get_dev_from_ce(dst_ce);
  if (src_dev == dst_dev) {
    send_local_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset],
        get_local_rank_from_ce(dst_ce));
  } else {
    send_remote_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset],
        dst_ce);
  }
}

__device__ void charm::send_do_term_msg_pe(int dst_local_rank) {
  // Prepare message
  if (threadIdx.x == 0) {
    envelope* env = create_envelope(msgtype::do_terminate, 0,
        (size_t&)s_mem[s_idx::offset]);
    s_mem[s_idx::env] = (uint64_t)env;
  }
  __syncthreads();

  // Send message to child PE
  send_local_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset],
      dst_local_rank);
}
