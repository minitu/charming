#ifndef _CHARE_H_
#define _CHARE_H_

#include <nvshmem.h>
#include "common.h"
#include "message.h"
#include "kernel.h"

// Maximum number of chare types
#define CHARE_TYPE_CNT_MAX 1024

// GPU constant memory
extern __constant__ int c_n_pes;

// GPU shared memory
extern __shared__ uint64_t s_mem[];

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

template <class C>
struct entry_method_base {
  __device__ virtual void operator()(C& chare, void* arg) = 0;
};

template <class C, void Func(C&, void*)>
struct entry_method : entry_method_base<C> {
  __device__ virtual void operator()(C& chare, void* arg) { Func(chare, arg); }
};

struct chare_proxy_base {
  int id;

  __device__ chare_proxy_base() {}
  __device__ virtual void create_local(int n_local, int n_total, int start_idx,
      int end_idx, void* map_ptr, void* obj_ptr) = 0;
  __device__ virtual void call(int idx, int ep, void* arg) = 0;
};

struct chare_proxy_table {
  chare_proxy_base* proxies[CHARE_TYPE_CNT_MAX];
  int count;

  chare_proxy_table() : count(0) {}
};

}; // namespace charm

// GPU global memory
extern __device__ __managed__ charm::chare_proxy_table* proxy_tables;

namespace charm {

template <typename C>
struct chare_proxy : chare_proxy_base {
  C** objects; // Chare objects on this PE (local chares)

  int n_local; // Number of local chares
  int n_total; // Total number of chares

  int start_idx; // Starting index of local chares
  int end_idx; // Ending index of local chares

  int* loc_map; // Chare-PE location map

  entry_method_base<C>** entry_methods; // Entry methods
  int em_count; // Number of registered entry methods

  __device__ chare_proxy(int n_em)
    : objects(nullptr), n_local(0), n_total(0), start_idx(-1), end_idx(-1),
      loc_map(nullptr), entry_methods(nullptr), em_count(0) {
    // Store this proxy for the runtime
    chare_proxy_table& my_proxy_table = proxy_tables[blockIdx.x];
    id = my_proxy_table.count++;
    assert(id < CHARE_TYPE_CNT_MAX);
    my_proxy_table.proxies[id] = this;

    // Allocate entry method table
    entry_methods = new entry_method_base<C>*[n_em];
  }

  // Add entry method to table
  template <void Func(C&, void*)>
  __device__ void add_entry_method() {
    entry_methods[em_count++] = new entry_method<C, Func>();
  }

  // Create local chares
  __device__ virtual void create_local(int n_local_, int n_total_, int start_idx_,
      int end_idx_, void* map_ptr, void* obj_ptr) {
    // Store information and create local chares
    if (threadIdx.x == 0) {
      n_local = n_local_;
      n_total = n_total_;
      objects = (n_local > 0) ? new C*[n_local] : nullptr;
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
    }
    __syncthreads();

    // On PEs other than 0, create location map using data from creation message
    if (map_ptr) {
      if (threadIdx.x == 0) {
        loc_map = new int[n_total];
        s_mem[s_idx::dst] = (uint64_t)loc_map;
        s_mem[s_idx::src] = (uint64_t)map_ptr;
        s_mem[s_idx::size] = (uint64_t)(sizeof(int) * n_total);
      }
      __syncthreads();
      memcpy_kernel((void*)s_mem[s_idx::dst], (void*)s_mem[s_idx::src],
          (size_t)s_mem[s_idx::size]);
    }
  }

  // Only called on PE 0
  // FIXME: Currently only block mapping
  __device__ void create(C& obj, int n) {
    // Divide the chares across all PEs
    int n_pes = c_n_pes;
    int my_pe = s_mem[s_idx::my_pe];
    int n_per_pe = n / n_pes;
    int rem = n % n_pes;

    // Create chare-PE location map
    if (threadIdx.x == 0) {
      loc_map = new int[n];
    }
    __syncthreads();
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
      if (threadIdx.x == 0) {
        for (int i = start_idx_; i <= end_idx_; i++) {
          loc_map[i] = pe;
        }
      }

      // Update start chare index
      start_idx_ += n_this;
      __syncthreads();
    }

    // Create chares
    // FIXME: Code duplication with above
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
        if (threadIdx.x == 0) {
          // Send creation messages to all other PEs
          // Payload includes location map and packed seed object
          size_t map_size = sizeof(int) * n;
          size_t obj_size = obj.pack_size();
          size_t payload_size = map_size + obj_size;
          size_t msg_size = envelope::alloc_size(sizeof(create_msg) + payload_size);

          // Create message
          envelope* env = create_envelope(msgtype::create, msg_size,
              (size_t*)&s_mem[s_idx::offset], pe);
          create_msg* msg = new ((char*)env + sizeof(envelope)) create_msg(id, n_this, n, start_idx_, end_idx_);
          printf("!!! env: %p, creation msg chare ID %d n_local %d n_total %d start_idx %d end_idx %d\n",
              env, msg->chare_id, msg->n_local, msg->n_total, msg->start_idx, msg->end_idx);
          s_mem[s_idx::env] = (uint64_t)env;
          s_mem[s_idx::msg_size] = (uint64_t)msg_size;

          // Prepare memcpy and pack seed object
          char* start = (char*)msg + sizeof(create_msg);
          s_mem[s_idx::dst] = (uint64_t)start;
          s_mem[s_idx::src] = (uint64_t)loc_map;
          s_mem[s_idx::size] = (uint64_t)map_size;
          start += map_size;
          obj.pack(start);
        }
        __syncthreads();

        // Copy message content
        memcpy_kernel((void*)s_mem[s_idx::dst], (void*)s_mem[s_idx::src],
            (size_t)s_mem[s_idx::size]);

        // Send creation message to target PE
        send_msg((envelope*)s_mem[s_idx::env], (size_t)s_mem[s_idx::offset],
            (size_t)s_mem[s_idx::msg_size], pe);
      }

      // Update start chare index
      start_idx_ += n_this;
      __syncthreads();
    }
  }

  __device__ virtual void call(int idx, int ep, void* arg) {
    assert(idx >= start_idx && idx <= end_idx);
    (*(entry_methods[ep]))(*(objects[idx - start_idx]), arg);
  }

  // Note: Chare should have been already created at this PE via a creation message
  inline __device__ void invoke(int idx, int ep) { invoke(idx, ep, nullptr, 0); }
  inline __device__ void invoke(int idx, int ep, void* buf, size_t size) {
    send_reg_msg(id, idx, ep, buf, size, loc_map[idx]);
  }
  inline __device__ void invoke(int idx, int ep, const message& msg) {
    send_user_msg(id, idx, ep, msg);
  }
  inline __device__ void invoke(int idx, int ep, const message& msg, size_t size) {
    send_user_msg(id, idx, ep, msg, size);
  }
  inline __device__ void invoke_all(int ep) { invoke_all(ep, nullptr, 0); }
  inline __device__ void invoke_all(int ep, void* buf, size_t size) {
    for (int i = 0; i < n_total; i++) {
      invoke(i, ep, buf, size);
    }
  }

  inline __device__ void alloc_msg(message& msg, int idx, size_t size) {
    size_t msg_size = envelope::alloc_size(sizeof(regular_msg) + size);
    msg.dst_pe = loc_map[idx];
    msg.env = create_envelope(msgtype::user, msg_size, &msg.offset, msg.dst_pe);
  }
  inline __device__ void free_msg(message& msg) { /*TODO*/ }
};

} // namespace charm

#endif // _CHARE_H_
