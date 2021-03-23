#ifndef _MESSAGE_H_
#define _MESSAGE_H_

enum class MsgType {
  Regular,
  Create,
  Terminate
};

struct Envelope {
  MsgType type;
  size_t size;
  int src_pe;

  __device__ Envelope(MsgType type_, size_t size_, int src_pe_)
    : type(type_), size(size_), src_pe(src_pe_) {}

  inline static __device__ size_t alloc_size(size_t size_) {
    return sizeof(Envelope) + size_;
  }
};

struct RegularMsg {
  int chare_id;
  int ep_id;

  __device__ RegularMsg(int chare_id_, int ep_id_)
    : chare_id(chare_id_), ep_id(ep_id_) {}
};

struct CreateMsg {
};

#endif // MESSAGE_H_
