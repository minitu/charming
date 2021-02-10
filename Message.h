#ifndef MESSAGE_H_
#define MESSAGE_H_

struct Message {
  int ep;
  int src_sm;
  void* data;
};

#endif // MESSAGE_H_
