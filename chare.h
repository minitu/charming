#ifndef _CHARE_H_
#define _CHARE_H_

#include <nvshmem.h>
#include <nvfunctional>
#include "scheduler.h"

namespace charm {

struct entry_method {
  int idx; // FIXME: Needed?

  __device__ entry_method(int idx_) : idx(idx_) {}
  __device__ virtual void call(void* chare) const = 0;
};

template <typename C, typename T>
struct entry_method_impl : entry_method {
  nvstd::function<T> fn;

  __device__ entry_method_impl(int idx, nvstd::function<T> fn_)
    : entry_method(idx), fn(fn_) {}
  __device__ virtual void call(void* chare) const { fn(*(C*)chare); }
};

struct chare_type {
  int id; // FIXME: Needed?

  __device__ chare_type(int id_) : id(id_) {}
  __device__ virtual void alloc() = 0;
  __device__ virtual void unpack(void* ptr) = 0;
  __device__ virtual void call(int ep) = 0;
};

template <typename C>
struct chare : chare_type {
  C* obj;
  entry_method** entry_methods;

  __device__ chare(int id_) : chare_type(id_), obj(nullptr), entry_methods(nullptr) {}
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
      size_t msg_size = envelope::alloc_size(sizeof(create_msg) + payload_size);
      envelope* env = create_envelope(msgtype::create, msg_size);

      create_msg* msg = new ((char*)env + sizeof(envelope)) create_msg(id);
      obj_.pack((char*)msg + sizeof(create_msg));

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

}

#endif // _CHARE_H_
