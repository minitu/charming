#ifndef _CHARE_H_
#define _CHARE_H_

#include <nvshmem.h>
#include <nvfunctional>
#include "scheduler.h"

struct EntryMethod {
  int idx; // FIXME: Needed?

  __device__ EntryMethod(int idx_) : idx(idx_) {}
  __device__ virtual void call(void* chare) const = 0;
};

template <typename C, typename T>
struct EntryMethodImpl : EntryMethod {
  nvstd::function<T> fn;

  __device__ EntryMethodImpl(int idx, nvstd::function<T> fn_)
    : EntryMethod(idx), fn(fn_) {}
  __device__ virtual void call(void* chare) const { fn(*(C*)chare); }
};

struct ChareType {
  int id; // FIXME: Needed?

  __device__ ChareType(int id_) : id(id_) {}
  __device__ virtual void alloc() = 0;
  __device__ virtual void unpack(void* ptr) = 0;
  __device__ virtual void call(int ep) = 0;
};

template <typename C>
struct Chare : ChareType {
  C* obj;
  EntryMethod** entry_methods;

  __device__ Chare(int id_) : ChareType(id_), obj(nullptr), entry_methods(nullptr) {}
  __device__ void alloc(C& obj_) { obj = new C(obj_); }
  __device__ virtual void alloc() { obj = new C; }
  __device__ virtual void unpack(void* ptr) { obj->unpack(ptr); }
  __device__ virtual void call(int ep) { entry_methods[ep]->call(obj); }

  // TODO: Currently 1 chare per PE
  __device__ void create(C& obj_) {
    // Create one object for myself (PE 0)
    alloc(obj_);

    // Send creation messages to all other PEs
    int my_pe = nvshmem_my_pe();
    for (int pe = 0; pe < nvshmem_n_pes(); pe++) {
      if (pe == my_pe) continue;

      size_t payload_size = obj_.pack_size();
      size_t msg_size = Envelope::alloc_size(sizeof(CreateMsg) + payload_size);
      Envelope* env = create_envelope(MsgType::Create, msg_size);

      CreateMsg* create_msg = new ((char*)env + sizeof(Envelope)) CreateMsg(id);
      obj_.pack((char*)create_msg + sizeof(CreateMsg));

      send_msg(env, msg_size, pe);
    }
  }

  // TODO
  // - Change to send to chare instead of PE
  // - Support entry method parameters (single buffer for now)
  // Note: Chare should have been already created at this PE via a creation message
  __device__ void invoke(int idx, int ep) {
    if (idx == -1) {
      // TODO: Broadcast to all chares
    } else {
      // Send a regular message to the target PE
      send_reg_msg(idx, ep, 0, idx);
    }
  }
};

#endif // _CHARE_H_
