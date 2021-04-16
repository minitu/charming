#ifndef _CHARE_H_
#define _CHARE_H_

#include <nvshmem.h>
#include <nvfunctional>
#include "scheduler.h"

namespace charm {

struct entry_method {
  int idx; // FIXME: Needed?

  __device__ entry_method(int idx_) : idx(idx_) {}
  __device__ virtual void call(void* chare, void* arg) const = 0;
};

template <typename C>
struct entry_method_impl : entry_method {
  nvstd::function<void(C&,void*)> fn;

  __device__ entry_method_impl(int idx, nvstd::function<void(C&,void*)> fn_)
    : entry_method(idx), fn(fn_) {}
  __device__ virtual void call(void* chare, void* arg) const { fn(*(C*)chare, arg); }
};

struct chare_type {
  int id;

  __device__ chare_type(int id_) : id(id_) {}
  __device__ virtual void alloc(int count, int start_idx, int end_idx) = 0;
  __device__ virtual void unpack(void* ptr, int idx) = 0;
  __device__ virtual void call(int idx, int ep, void* arg) = 0;
};

template <typename C>
struct chare : chare_type {
  C** objects;
  int start_idx;
  int end_idx;
  entry_method** entry_methods;

  __device__ chare(int id_)
    : chare_type(id_), objects(nullptr), start_idx(-1),
      end_idx(-1), entry_methods(nullptr) {}

  __device__ virtual void alloc(int count, int start_idx_, int end_idx_) {
    objects = new C*[count];
    start_idx = start_idx_;
    end_idx = end_idx_;
  }

  __device__ void set(C& obj, int idx) { objects[idx] = new C(obj); }

  __device__ virtual void unpack(void* ptr, int idx) {
    objects[idx] = new C;
    objects[idx]->unpack(ptr);
  }

  __device__ virtual void call(int idx, int ep, void* arg) {
    assert(idx >= start_idx && idx <= end_idx);
    entry_methods[ep]->call(objects[idx - start_idx], arg);
  }

  __device__ void create(C& obj, int n) {
    // Divide the chares across all PEs
    int n_pes = nvshmem_n_pes();
    int my_pe = nvshmem_my_pe();
    int n_per_pe = n / n_pes;
    int rem = n % n_pes;

    // Create chares
    // TODO: Currently block mapping
    int n_this = -1;
    int start_idx_ = 0;
    int end_idx_ = 0;
    for (int pe = 0; pe < n_pes; pe++) {
      // Figure out number of chares for this PE
      n_this = n_per_pe;
      if (pe < rem) n_this++;

      // Update end chare index
      end_idx_ = start_idx_ + n_this - 1;

      // Create chares for this PE
      if (pe == my_pe) {
        alloc(n_this);
        for (int i = 0; i < n_this; i++) {
          set(obj, i);
        }

        // Store chare index range
        start_idx = start_idx_;
        end_idx = end_idx_;
      } else {
        // Send creation messages to all other PEs
        size_t payload_size = obj.pack_size();
        size_t msg_size = envelope::alloc_size(sizeof(create_msg) + payload_size);
        envelope* env = create_envelope(msgtype::create, msg_size);
        create_msg* msg = new ((char*)env + sizeof(envelope)) create_msg(id, n_this, start_idx_, end_idx_);
        obj.pack((char*)msg + sizeof(create_msg));

        send_msg(env, msg_size, pe);
      }

      // Update start chare index
      start_idx_ += n_this;
    }
  }

  // TODO
  // - Change to send to chare instead of PE
  // Note: Chare should have been already created at this PE via a creation message
  inline __device__ void invoke(int idx, int ep) { invoke(idx, ep, nullptr, 0); }
  __device__ void invoke(int idx, int ep, void* buf, size_t size) {
    if (idx == -1) {
      // TODO: Broadcast to all chares
    } else {
      // Send a regular message to the target PE
      // TODO: Need to figure out which PE to send it to
      send_reg_msg(id, idx, ep, buf, size, idx);
    }
  }
};

}

#endif // _CHARE_H_
