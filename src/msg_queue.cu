#include "msg_queue.h"
#include <assert.h>
#include <nvshmem.h>

MsgQueueMetaShell::MsgQueueMetaShell(size_t space_) {
  space = space_;
  meta = static_cast<MsgQueueMeta*>(nvshmem_malloc(sizeof(MsgQueueMeta)));
  assert(meta);
}

MsgQueueMetaShell::~MsgQueueMetaShell() {
  nvshmem_free(meta);
}

__device__ void MsgQueueMetaShell::init() {
  assert(meta);
  meta->read = 0;
  meta->write = 0;
  meta->watermark = static_cast<offset_t>(space);
}

MsgQueueShell::MsgQueueShell(size_t space_) {
  space = space_;
  queue = static_cast<void*>(nvshmem_malloc(space));
  assert(queue);
}

MsgQueueShell::~MsgQueueShell() {
  nvshmem_free(queue);
}

__device__ void* MsgQueueShell::addr(offset_t offset) {
  return static_cast<void*>(static_cast<char*>(queue) + offset);
}
