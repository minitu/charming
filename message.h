#ifndef _MESSAGE_H_
#define _MESSAGE_H_

namespace charm {

enum class msgtype {
  regular,
  create,
  terminate
};

struct envelope {
  msgtype type;
  size_t size;
  int src_pe;

  __device__ envelope(msgtype type_, size_t size_, int src_pe_)
    : type(type_), size(size_), src_pe(src_pe_) {}

  inline static __device__ size_t alloc_size(size_t size_) {
    return sizeof(envelope) + size_;
  }
};

struct regular_msg {
  int chare_id;
  int ep_id;

  __device__ regular_msg(int chare_id_, int ep_id_)
    : chare_id(chare_id_), ep_id(ep_id_) {}
};

struct create_msg {
  int chare_id;

  __device__ create_msg(int chare_id_) : chare_id(chare_id_) {}
};

}

#endif // _MESSAGE_H_
