#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#include <stdio.h>

#define ALIGN_SIZE 16

namespace charm {

enum class msgtype {
  regular,
  create,
  terminate
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
  alignas(ALIGN_SIZE) int n_chares;
  alignas(ALIGN_SIZE) int start_idx;
  alignas(ALIGN_SIZE) int end_idx;

  __device__ create_msg(int chare_id_, int n_chares_, int start_idx_, int end_idx_)
    : chare_id(chare_id_), n_chares(n_chares_), start_idx(start_idx_), end_idx(end_idx_) {}
};

}

#endif // _MESSAGE_H_
