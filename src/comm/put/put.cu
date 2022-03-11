#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "charming.h"
#include "message.h"
#include "comm.h"
#include "scheduler.h"
#include "msg_queue.h"
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
__device__ MsgQueueMetaShell* recv_meta_shell;
__device__ MsgQueueMetaShell* send_meta_shell;
__device__ MsgQueueShell* msg_queue_shell;
__device__ size_t msg_queue_size;
__device__ spsc_ringbuf_t* mbuf;
__device__ size_t mbuf_size;

// Host memory
MsgQueueMetaShell* h_recv_meta_shell;
MsgQueueMetaShell* h_send_meta_shell;
MsgQueueShell* h_msg_queue_shell;
size_t* h_msg_queue_size;
MsgQueueMetaShell* d_recv_meta_shell;
MsgQueueMetaShell* d_send_meta_shell;
MsgQueueShell* d_msg_queue_shell;
spsc_ringbuf_t* h_mbuf;
size_t* h_mbuf_size;

void charm::comm_init_host(int n_pes) {
  // Allocate message queue with NVSHMEM
  cudaMallocHost(&h_recv_meta_shell, sizeof(MsgQueueMetaShell) * n_pes);
  cudaMallocHost(&h_send_meta_shell, sizeof(MsgQueueMetaShell) * n_pes);
  cudaMallocHost(&h_msg_queue_shell, sizeof(MsgQueueShell) * n_pes);
  cudaMallocHost(&h_msg_queue_size, sizeof(size_t));
  *h_msg_queue_size = (1 << 28);
  for (int i = 0; i < n_pes; i++) {
    new (&h_recv_meta_shell[i]) MsgQueueMetaShell(*h_msg_queue_size);
    new (&h_send_meta_shell[i]) MsgQueueMetaShell(*h_msg_queue_size);
    new (&h_msg_queue_shell[i]) MsgQueueShell(*h_msg_queue_size);
  }
  cudaMalloc(&d_recv_meta_shell, sizeof(MsgQueueMetaShell) * n_pes);
  cudaMalloc(&d_send_meta_shell, sizeof(MsgQueueMetaShell) * n_pes);
  cudaMalloc(&d_msg_queue_shell, sizeof(MsgQueueShell) * n_pes);
  cudaMallocHost(&h_mbuf_size, sizeof(size_t));
  *h_mbuf_size = *h_msg_queue_size;
  h_mbuf = spsc_ringbuf_malloc(*h_mbuf_size);

  cudaMemcpyAsync(d_recv_meta_shell, h_recv_meta_shell,
      sizeof(MsgQueueMetaShell) * n_pes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_send_meta_shell, h_send_meta_shell,
      sizeof(MsgQueueMetaShell) * n_pes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_msg_queue_shell, h_msg_queue_shell,
      sizeof(MsgQueueShell) * n_pes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(recv_meta_shell, &d_recv_meta_shell,
      sizeof(MsgQueueMetaShell*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(send_meta_shell, &d_send_meta_shell,
      sizeof(MsgQueueMetaShell*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(msg_queue_shell, &d_msg_queue_shell,
      sizeof(MsgQueueShell*), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(msg_queue_size, h_msg_queue_size,
      sizeof(size_t), 0, cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf, &h_mbuf, sizeof(spsc_ringbuf_t*), 0,
      cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(mbuf_size, h_mbuf_size, sizeof(size_t), 0,
      cudaMemcpyHostToDevice, stream);
}

void charm::comm_fini_host(int n_pes) {
  for (int i = 0; i < n_pes; i++) {
    // Explicitly call destructors since these were placement new'ed
    h_recv_meta_shell[i].MsgQueueMetaShell::~MsgQueueMetaShell();
    h_send_meta_shell[i].MsgQueueMetaShell::~MsgQueueMetaShell();
    h_msg_queue_shell[i].MsgQueueShell::~MsgQueueShell();
  }
  cudaFreeHost(h_recv_meta_shell);
  cudaFreeHost(h_send_meta_shell);
  cudaFreeHost(h_msg_queue_shell);
  cudaFreeHost(h_msg_queue_size);
  cudaFree(d_recv_meta_shell);
  cudaFree(d_send_meta_shell);
  cudaFree(d_msg_queue_shell);
  cudaFreeHost(h_mbuf_size);
  spsc_ringbuf_free(h_mbuf);
}

__device__ charm::comm::comm() {
  begin_term_flag = false;
  do_term_flag = false;

  for (int i = 0; i < c_n_pes; i++) {
    recv_meta_shell[i].init();
    send_meta_shell[i].init();
  }
  spsc_ringbuf_init(mbuf, mbuf_size);
}

__device__ void charm::comm::process_local() {}

__device__ void charm::comm::process_remote() {
  // Retrieve symmetric addresses for metadata
  for (int src_pe = 0; src_pe < c_n_pes; src_pe++) {
    MsgQueueMeta* recv_meta = recv_meta_shell[src_pe].meta;
    offset_t write = nvshmem_longlong_atomic_fetch(&recv_meta->write, c_my_pe);
    PDEBUG("PE %d: checking msg queue from PE %d, read %lld, write %lld\n",
        c_my_pe, src_pe, recv_meta->read, write);
    if (recv_meta->read < write) {
      // There are messages to process
      MsgQueueShell& msgq_shell = msg_queue_shell[src_pe];
      ssize_t msg_size = process_msg(msgq_shell.addr(recv_meta->read),
          begin_term_flag, do_term_flag);
      recv_meta->read += msg_size;
    }
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

  // TODO: Don't assume no wrap-around, use watermark
  // Retrieve symmetric addresses for metadata
  MsgQueueMeta* send_meta = send_meta_shell[dst_pe].meta;
  MsgQueueMeta* recv_meta = recv_meta_shell[c_my_pe].meta;
  offset_t avail = msg_queue_size - send_meta->write;
  assert(avail >= msg_size);

  // Send message
  MsgQueueShell& msgq_shell = msg_queue_shell[c_my_pe];
  nvshmem_char_put((char*)msgq_shell.addr(send_meta->write), (char*)env, env->size, dst_pe);
  send_meta->write += msg_size;

  // FIXME: Don't update receiver's write offset every time?
  nvshmem_fence();
  nvshmem_longlong_atomic_set(&recv_meta->write, send_meta->write, dst_pe);

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

