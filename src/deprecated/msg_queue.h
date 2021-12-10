#ifndef _MSG_QUEUE_H_
#define _MSG_QUEUE_H_

/* A ring buffer implementation of message queue.
 * Reference: https://ferrous-systems.com/blog/lock-free-ring-buffer/
 */

typedef long long offset_t;

struct MsgQueueMeta {
  offset_t read;
  offset_t write;
  offset_t watermark;
};

struct MsgQueueMetaShell {
  size_t space;
  MsgQueueMeta* meta;

  MsgQueueMetaShell(size_t space_);
  ~MsgQueueMetaShell();

  __device__ void init();
};

struct MsgQueueShell {
  size_t space;
  void* queue;

  MsgQueueShell(size_t space_);
  ~MsgQueueShell();

  __device__ void* addr(offset_t offset);
};

#endif // _MSG_QUEUE_H_
