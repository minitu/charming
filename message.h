#ifndef _MESSAGE_H_
#define _MESSAGE_H_

struct Message {
  size_t size;
  int chare_id;
  int ep_id;

  __device__ Message(size_t payload_size, int chare_id_, int ep_id_)
    : size(sizeof(Message) + payload_size), chare_id(chare_id_), ep_id(ep_id_) {}
  inline static __device__ size_t alloc_size(size_t size_) { return sizeof(Message) + size_; }
};

#endif // MESSAGE_H_
