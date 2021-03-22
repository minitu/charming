#ifndef _MESSAGE_H_
#define _MESSAGE_H_

enum class MsgType {
  Regular,
  Create,
  Terminate
};

struct Message {
  MsgType type;
  int chare_id;
  int ep_id;
  size_t size;

  __device__ Message(MsgType type_, int chare_id_, int ep_id_, size_t payload_size)
    : type(type_), chare_id(chare_id_), ep_id(ep_id_), size(sizeof(Message) + payload_size) {}

  inline static __device__ size_t alloc_size(size_t size_) { return sizeof(Message) + size_; }
};

#endif // MESSAGE_H_
