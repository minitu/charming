#ifndef _CHARE_H_
#define _CHARE_H_

#include <nvshmem.h>
#include <nvfunctional>
#include "message.h"

namespace charm {

struct chare {
  int i;
  int n;

  __device__ chare() : i(-1), n(0) {}

  // Packing/unpacking functions to be overloaded by the user
  __device__ size_t pack_size() { return 0; }
  __device__ void pack(void* ptr) {}
  __device__ void unpack(void* ptr) {}
};

template <typename C>
struct entry_method {
  nvstd::function<void(C&,void*)> fn;

  __device__ entry_method(nvstd::function<void(C&,void*)> fn_) : fn(fn_) {}
  __device__ virtual void call(void* chare, void* arg) const { fn(*(C*)chare, arg); }
};

struct chare_proxy_base {
  int id;

  __device__ chare_proxy_base() {}
  __device__ virtual void create_local(int n_local, int n_total, int start_idx,
      int end_idx, void* map_ptr, void* obj_ptr) = 0;
  __device__ virtual void call(int idx, int ep, void* arg) = 0;
};

}; // namespace charm

extern __device__ charm::chare_proxy_base* chare_proxies[];
extern __device__ int chare_proxy_cnt;

namespace charm {

template <typename C>
struct chare_proxy : chare_proxy_base {
  C** objects; // Chare objects on this PE (local chares)

  int n_local; // Number of local chares
  int n_total; // Total number of chares

  int start_idx; // Starting index of local chares
  int end_idx; // Ending index of local chares

  int* loc_map; // Chare-PE location map

  entry_method<C>** entry_methods; // Entry methods
  int em_count; // Number of registered entry methods

  __device__ chare_proxy(int n_em)
    : objects(nullptr), n_local(0), n_total(0), start_idx(-1), end_idx(-1),
      loc_map(nullptr), entry_methods(nullptr), em_count(0) {
    // Store this proxy for the runtime
    id = chare_proxy_cnt;
    chare_proxies[chare_proxy_cnt++] = this;

    // Allocate entry method table
    entry_methods = new entry_method<C>*[n_em];
  }

  // Add entry method to table
  __device__ void add_entry_method(const nvstd::function<void(C&,void*)>& fn) {
    entry_methods[em_count++] = new entry_method<C>(fn);
  }

  // Create local chares
  __device__ virtual void create_local(int n_local_, int n_total_, int start_idx_,
      int end_idx_, void* map_ptr, void* obj_ptr) {
    n_local = n_local_;
    n_total = n_total_;
    objects = new C*[n_local];
    start_idx = start_idx_;
    end_idx = end_idx_;

    // Create chares and unpack
    // Make chare index and number of chares accessible to the user
    for (int idx = 0; idx < n_local; idx++) {
      objects[idx] = new C;
      objects[idx]->unpack(obj_ptr);
      objects[idx]->i = start_idx + idx;
      objects[idx]->n = n_total;
    }

    // Create location map for PEs other than PE 0
    if (map_ptr) {
      loc_map = new int[n_total];
      memcpy(loc_map, map_ptr, sizeof(int) * n_total);
    }
  }

  // Only called on PE 0
  // FIXME: Currently only block mapping
  __device__ void create(C& obj, int n) {
    // Divide the chares across all PEs
    int n_pes = nvshmem_n_pes();
    int my_pe = nvshmem_my_pe();
    int n_per_pe = n / n_pes;
    int rem = n % n_pes;

    // Create chare-PE location map
    loc_map = new int[n];
    int n_this = -1;
    int start_idx_ = 0;
    int end_idx_ = 0;
    for (int pe = 0; pe < n_pes; pe++) {
      // Figure out number of chares for this PE
      n_this = n_per_pe;
      if (pe < rem) n_this++;

      // Update end chare index
      end_idx_ = start_idx_ + n_this - 1;

      // Fill in location map
      for (int i = start_idx_; i <= end_idx_; i++) {
        loc_map[i] = pe;
      }

      // Update start chare index
      start_idx_ += n_this;
    }

    // Create chares
    n_this = -1;
    start_idx_ = 0;
    end_idx_ = 0;
    for (int pe = 0; pe < n_pes; pe++) {
      // Figure out number of chares for this PE
      n_this = n_per_pe;
      if (pe < rem) n_this++;

      // Update end chare index
      end_idx_ = start_idx_ + n_this - 1;

      if (pe == my_pe) {
        // Create chares for this PE
        create_local(n_this, n, start_idx_, end_idx_, nullptr, &obj);
      } else {
        // Send creation messages to all other PEs
        // Payload includes location map and packed seed object
        size_t map_size = sizeof(int) * n;
        size_t obj_size = obj.pack_size();
        size_t payload_size = map_size + obj_size;
        size_t msg_size = envelope::alloc_size(sizeof(create_msg) + payload_size);
        size_t offset;

        // Create message
        envelope* env = create_envelope(msgtype::create, msg_size, &offset);
        create_msg* msg = new ((char*)env + sizeof(envelope)) create_msg(id, n_this, n, start_idx_, end_idx_);

        // Pack location map and seed object
        char* tmp = (char*)msg + sizeof(create_msg);
        memcpy(tmp, loc_map, map_size);
        tmp += map_size;
        obj.pack(tmp);

        // Send creation message to target PE
        send_msg(env, offset, msg_size, pe);
      }

      // Update start chare index
      start_idx_ += n_this;
    }
  }

  __device__ virtual void call(int idx, int ep, void* arg) {
    assert(idx >= start_idx && idx <= end_idx);
    entry_methods[ep]->call(objects[idx - start_idx], arg);
  }

  // Note: Chare should have been already created at this PE via a creation message
  inline __device__ void invoke(int idx, int ep) { invoke(idx, ep, nullptr, 0); }
  inline __device__ void invoke(int idx, int ep, void* buf, size_t size) {
    // Send a regular message to the target PE
    send_reg_msg(id, idx, ep, buf, size, loc_map[idx]);
  }
  inline __device__ void invoke_all(int ep) { invoke_all(ep, nullptr, 0); }
  inline __device__ void invoke_all(int ep, void* buf, size_t size) {
    for (int i = 0; i < n_total; i++) {
      invoke(i, ep, buf, size);
    }
  }
};

} // namespace charm

#endif // _CHARE_H_
