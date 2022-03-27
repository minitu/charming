#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "ringbuf.h"
#include "util.h"

using namespace charm;

extern cudaStream_t stream;

// GPU constant memory
extern __constant__ int c_my_pe;
extern __constant__ int c_n_pes;
extern __constant__ int c_my_pe_node;
extern __constant__ int c_n_pes_node;
extern __constant__ int c_n_nodes;

// GPU global memory
__device__ mpsc_ringbuf_t* rbuf;
__device__ size_t rbuf_size;
__device__ spsc_ringbuf_t* mbuf;
__device__ size_t mbuf_size;

// Host memory
mpsc_ringbuf_t* h_rbuf;
spsc_ringbuf_t* h_mbuf;

void charm::comm_init_host(int n_pes) {
  // Allocate message queue with NVSHMEM
  size_t h_rbuf_size = (1 << 28);
  h_rbuf = mpsc_ringbuf_malloc(h_rbuf_size);
  size_t h_mbuf_size = (1 << 28);
  h_mbuf = spsc_ringbuf_malloc(h_mbuf_size);

  cudaMemcpyToSymbolAsync(rbuf, &h_rbuf, sizeof(mpsc_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(rbuf_size, &h_rbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf, sizeof(spsc_ringbuf_t*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, &h_mbuf_size, sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
}

void charm::comm_fini_host(int n_pes) {
  spsc_ringbuf_free(h_mbuf);
  mpsc_ringbuf_free(h_rbuf);
}

__device__ void charm::comm::init() {
  begin_term_flag = false;
  do_term_flag = false;

  mpsc_ringbuf_init(rbuf, rbuf_size);
  spsc_ringbuf_init(mbuf, mbuf_size);
}

__device__ void charm::comm::process_local() {}

__device__ void charm::comm::process_remote() {
  size_t len, off;
  if ((len = mpsc_ringbuf_consume(rbuf, &off)) != 0) {
    // Retrieved a contiguous range, there could be multiple messages
    size_t rem = len;
    ssize_t msg_size;
    while (rem) {
      process_msg(rbuf->addr(off), &msg_size, begin_term_flag, do_term_flag);
      off += msg_size;
      rem -= msg_size;
    }
    mpsc_ringbuf_release(rbuf, len);
  }
}

__device__ void charm::comm::cleanup() {}

__device__ envelope* charm::create_envelope(msgtype type, size_t msg_size, size_t* offset) {
  // Reserve space for this message in message buffer
  ringbuf_off_t mbuf_off = spsc_ringbuf_acquire(mbuf, msg_size);
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
