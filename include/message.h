#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#define ALIGN_SIZE 16

namespace charm {

enum class msgtype {
  regular,
  create,
  begin_terminate,
  do_terminate
};

// TODO: Is alignment too aggressive? Otherwise we observe segfaults
struct alignas(ALIGN_SIZE) envelope {
  alignas(ALIGN_SIZE) msgtype type;
  alignas(ALIGN_SIZE) size_t size;
  alignas(ALIGN_SIZE) int src_pe;

  __device__ envelope(msgtype type_, size_t size_, int src_pe_)
    : type(type_), size(size_), src_pe(src_pe_) {}

  inline static __device__ size_t alloc_size(size_t size_) {
    // Need to satisfy alignment
    size_t s = sizeof(envelope) + size_;
    size_t rem = s % ALIGN_SIZE;
    if (rem != 0) s = s + ALIGN_SIZE - rem;

    return s;
  }
};

struct alignas(ALIGN_SIZE) regular_msg {
  alignas(ALIGN_SIZE) int chare_id;
  alignas(ALIGN_SIZE) int chare_idx;
  alignas(ALIGN_SIZE) int ep_id;

  __device__ regular_msg(int chare_id_, int chare_idx_, int ep_id_)
    : chare_id(chare_id_), chare_idx(chare_idx_), ep_id(ep_id_) {}
};

struct alignas(ALIGN_SIZE) create_msg {
  alignas(ALIGN_SIZE) int chare_id;
  alignas(ALIGN_SIZE) int n_local;
  alignas(ALIGN_SIZE) int n_total;
  alignas(ALIGN_SIZE) int start_idx;
  alignas(ALIGN_SIZE) int end_idx;

  __device__ create_msg(int chare_id_, int n_local_, int n_total_, int start_idx_, int end_idx_)
    : chare_id(chare_id_), n_local(n_local_), n_total(n_total_), start_idx(start_idx_), end_idx(end_idx_) {}
};

__device__ envelope* create_envelope(msgtype type, size_t msg_size, size_t* offset);
__device__ void send_msg(envelope* env, size_t offset, size_t msg_size, int dst_pe);
__device__ void send_reg_msg(int chare_id, int chare_idx, int ep_id, void* buf, size_t payload_size, int dst_pe);
__device__ void send_begin_term_msg(int dst_pe);
__device__ void send_do_term_msg(int dst_pe);

}

#endif // _MESSAGE_H_
