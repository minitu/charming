#ifndef _MESSAGE_H_
#define _MESSAGE_H_

// Alignment needed to avoid misaligned address errors
#define ALIGN_SIZE 16

namespace charm {

// Message types
enum class msgtype : int {
  regular,
  request,
  forward,
  user,
  begin_terminate,
  do_terminate
};

// Regular message header between chares
struct alignas(ALIGN_SIZE) regular_msg {
  int chare_id;
  int chare_idx;
  int ep_id;
  int refnum;

  __device__ regular_msg(int chare_id_, int chare_idx_, int ep_id_, int refnum_)
    : chare_id(chare_id_), chare_idx(chare_idx_), ep_id(ep_id_), refnum(refnum_) {}
};

// Request sent from PE to CE
struct alignas(ALIGN_SIZE) request_msg : regular_msg {
  msgtype type;
  void* buf;
  size_t payload_size;
  int dst_pe;

  __device__ request_msg(int chare_id_, int chare_idx_, int ep_id_, msgtype type_,
      void* buf_, size_t payload_size_, int dst_pe_, int refnum_)
    : regular_msg(chare_id_, chare_idx_, ep_id_, refnum_), type(type_),
    buf(buf_), payload_size(payload_size_), dst_pe(dst_pe_) {}
};

// Message sent between CEs and forwarded to target PE
struct alignas(ALIGN_SIZE) forward_msg : regular_msg {
  int dst_pe;

  __device__ forward_msg(int chare_id_, int chare_idx_, int ep_id_, int dst_pe_,
      int refnum_)
    : regular_msg(chare_id_, chare_idx_, ep_id_, refnum_), dst_pe(dst_pe_) {}
};

// TODO: Is alignment too aggressive? Otherwise we observe segfaults
struct alignas(ALIGN_SIZE) envelope {
  msgtype type;
  size_t size;

  __device__ envelope(msgtype type_, size_t size_)
    : type(type_), size(size_) {}

  // Determine total size of message for memory allocation
  inline static __device__ size_t alloc_size(msgtype type, size_t payload_size) {
    size_t type_size = 0;
    if (type == msgtype::regular || type == msgtype::user) {
      type_size += sizeof(regular_msg);
    } else if (type == msgtype::request) {
      type_size += sizeof(request_msg);
    } else if (type == msgtype::forward) {
      type_size += sizeof(forward_msg);
    } else if (type != msgtype::begin_terminate
        && type != msgtype::do_terminate) {
      // TODO: User message API
      assert(false);
    }

    // Need to satisfy alignment
    size_t s = sizeof(envelope) + type_size + payload_size;
    size_t rem = s % ALIGN_SIZE;
    if (rem != 0) s = s + ALIGN_SIZE - rem;

    return s;
  }
};

// User Message API
struct alignas(ALIGN_SIZE) message {
  envelope* env;
  size_t offset;
  int dst_pe;

  __device__ message() : env(nullptr), offset(0), dst_pe(-1) {}
  /*
  __device__ void alloc(size_t size);
  __device__ void free();
  */
};

__device__ envelope* create_envelope(msgtype type, size_t payload_size,
    size_t& offset);
__device__ void send_local_msg(envelope* env, size_t offset, int dst_local_rank);
__device__ void send_remote_msg(envelope* env, size_t offset, int dst_pe);
__device__ void send_reg_msg(int chare_id, int chare_idx, int ep_id, void* buf,
    size_t payload_size, int dst_pe, int refnum);
__device__ void send_delegate_msg(request_msg* req);
__device__ void send_user_msg(int chare_id, int chare_idx, int ep_id,
    const message& msg);
__device__ void send_user_msg(int chare_id, int chare_idx, int ep_id,
    const message& msg, size_t payload_size);
__device__ void send_begin_term_msg();
__device__ void send_do_term_msg_ce(int dst_ce);
__device__ void send_do_term_msg_pe(int dst_local_rank);
__device__ void revive_mismatches(int chare_id, int chare_idx, int refnum);

}

#endif // _MESSAGE_H_
