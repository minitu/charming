#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#define ALIGN_SIZE 16

namespace charm {

// Message types
enum class msgtype : int {
  regular,
  begin_terminate,
  do_terminate,
  user,
  request
};

// Regular message header between chares
struct alignas(ALIGN_SIZE) regular_msg {
  alignas(ALIGN_SIZE) int chare_id;
  alignas(ALIGN_SIZE) int chare_idx;
  alignas(ALIGN_SIZE) int ep_id;

  __device__ regular_msg(int chare_id_, int chare_idx_, int ep_id_)
    : chare_id(chare_id_), chare_idx(chare_idx_), ep_id(ep_id_) {}
};

// Request sent from worker TB to comm TB
struct alignas(ALIGN_SIZE) request_msg : regular_msg {
  alignas(ALIGN_SIZE) msgtype type;
  alignas(ALIGN_SIZE) void* buf;
  alignas(ALIGN_SIZE) size_t payload_size;
  alignas(ALIGN_SIZE) int dst_pe;
};

// TODO: Is alignment too aggressive? Otherwise we observe segfaults
struct alignas(ALIGN_SIZE) envelope {
  alignas(ALIGN_SIZE) msgtype type;
  alignas(ALIGN_SIZE) size_t size;
  alignas(ALIGN_SIZE) int src_pe;

  __device__ envelope(msgtype type_, size_t size_, int src_pe_)
    : type(type_), size(size_), src_pe(src_pe_) {}

  // Determine total size of message for memory allocation
  inline static __device__ size_t alloc_size(msgtype type, size_t payload_size) {
    size_t type_size = 0;
    if (type == msgtype::regular || type == msgtype::user) {
      type_size += sizeof(regular_msg);
    } else if (type == msgtype::request) {
      type_size += sizeof(request_msg);
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

__device__ envelope* create_envelope(msgtype type, size_t msg_size,
    size_t& offset, int dst_pe);
__device__ void send_msg(envelope* env, size_t offset, int dst_pe);
__device__ void send_reg_msg(int chare_id, int chare_idx, int ep_id, void* buf,
    size_t payload_size, int dst_pe);
__device__ void send_user_msg(int chare_id, int chare_idx, int ep_id,
    const message& msg);
__device__ void send_user_msg(int chare_id, int chare_idx, int ep_id,
    const message& msg, size_t payload_size);
__device__ void send_term_msg(bool begin, int dst_pe);

}

#endif // _MESSAGE_H_
