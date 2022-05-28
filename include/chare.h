#ifndef _CHARE_H_
#define _CHARE_H_

#include <nvshmem.h>
#include "common.h"
#include "message.h"
#include "kernel.h"

#define CHARE_TYPE_CNT_MAX 1024 // Maximum number of chare types
#define EM_CNT_MAX 128 // Maximum number of entry methods per chare type
#define MISMATCH_MAX 128 // Maximum number of mismatched messages per chare type

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
  /*
  __device__ size_t pack_size() { return 0; }
  __device__ void pack(void* ptr) {}
  __device__ void unpack(void* ptr) {}
  */
};

template <class C>
struct entry_method_base {
  __device__ virtual void operator()(C& chare, void* arg) = 0;
};

template <class C, void Func(C&, void*)>
struct entry_method : entry_method_base<C> {
  __device__ virtual void operator()(C& chare, void* arg) { Func(chare, arg); }
};

struct mismatch_t {
  int msg_idx;
  uint64_t comp;
  int chare_idx;
  int refnum;

  __device__ mismatch_t() : msg_idx(-1) {}
};

struct chare_proxy_base {
  int id; // Chare array ID

  int n_local; // Number of local chares
  int n_total; // Total number of chares
  int start_idx; // Starting index of local chares
  int end_idx; // Ending index of local chares

  int* loc_map; // Chare-PE location map

  int em_count; // Number of registered entry methods

  int *refnum; // Reference numbers
  mismatch_t* mismatches;

  __device__ chare_proxy_base() : id(-1), n_local(0), n_total(0), start_idx(-1),
  end_idx(-1), loc_map(nullptr), em_count(0), refnum(nullptr), mismatches(nullptr)  {}
  __device__ virtual bool call(int idx, int ep, void* arg, int ref) = 0;
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
  entry_method_base<C>** entry_methods; // Entry methods

  __device__ chare_proxy()
    : chare_proxy_base(), objects(nullptr), entry_methods(nullptr) {
    // Store this proxy for the runtime
    chare_proxy_table& my_proxy_table = proxy_tables[blockIdx.x];
    id = my_proxy_table.count++;
    assert(id < CHARE_TYPE_CNT_MAX);
    my_proxy_table.proxies[id] = this;

    // Allocate entry method table
    entry_methods = new entry_method_base<C>*[EM_CNT_MAX];
  }

  // Add entry method to table
  template <void Func(C&, void*)>
  __device__ void add_entry_method() {
    entry_methods[em_count++] = new entry_method<C, Func>();
  }

  // Called on all TBs before main (single-threaded)
  __device__ void create(int n_chares, int* block_map) {
    int n_pes = c_n_pes;
    int my_pe = s_mem[s_idx::my_pe];

    // Allocate space for location map
    loc_map = new int[n_chares];

    // Default block mapping
    int n_chares_pe = n_chares / n_pes;
    int rem = n_chares % n_pes;
    int n_chares_cur = -1;
    int start = 0;
    int end = 0;
    for (int pe = 0; pe < n_pes; pe++) {
      // Calculate number of chares for this PE
      n_chares_cur = block_map ? block_map[pe] : n_chares_pe;
      if (!block_map && pe < rem) n_chares_cur++;

      // Update end chare index
      end = start + n_chares_cur - 1;

      // Store info for my PE and create local chare objects
      if (pe == my_pe) {
        n_local = n_chares_cur;
        n_total = n_chares;
        start_idx = start;
        end_idx = end;

        objects = (n_local > 0) ? new C*[n_local] : nullptr;
        refnum = (n_local > 0) ? new int[n_local] : nullptr;
        mismatches = (n_local > 0) ? new mismatch_t[MISMATCH_MAX] : nullptr;
        for (int idx = 0; idx < n_local; idx++) {
          objects[idx] = new C;
          objects[idx]->i = start_idx + idx;
          objects[idx]->n = n_total;
          refnum[idx] = -1;
        }
      }

      // Fill in location map
      for (int i = start; i <= end; i++) {
        loc_map[i] = pe;
      }

      // Update start chare index
      start += n_chares_cur;
    }
  }

  inline __device__ void create(int n_chares) { create(n_chares, nullptr); }

  __device__ virtual bool call(int idx, int ep, void* arg, int ref) {
    assert(idx >= start_idx && idx <= end_idx);
    int local_idx = idx - start_idx;

    // Don't execute the entry method if reference number doesn't match
    // (unless it is -1)
    if (ref != -1 && ref != refnum[local_idx]) {
      return false;
    }

    // Execute entry method
    (*(entry_methods[ep]))(*(objects[local_idx]), arg);

    return true;
  }

  inline __device__ void invoke(int idx, int ep) { invoke(idx, ep, nullptr, 0); }
  inline __device__ void invoke(int idx, int ep, void* buf, size_t size) {
    send_reg_msg(id, idx, ep, buf, size, loc_map[idx], -1);
  }
  inline __device__ void invoke(int idx, int ep, void* buf, size_t size, int refnum) {
    send_reg_msg(id, idx, ep, buf, size, loc_map[idx], refnum);
  }

  inline __device__ void invoke_all(int ep) { invoke_all(ep, nullptr, 0); }
  inline __device__ void invoke_all(int ep, void* buf, size_t size) {
    for (int i = 0; i < n_total; i++) {
      invoke(i, ep, buf, size);
    }
  }

  inline __device__ void invoke(int idx, int ep, const message& msg) {
    send_user_msg(id, idx, ep, msg);
  }
  inline __device__ void invoke(int idx, int ep, const message& msg, size_t size) {
    send_user_msg(id, idx, ep, msg, size);
  }

  // Single-threaded
  inline __device__ void set_refnum(int idx, int val) {
    assert(idx >= start_idx && idx <= end_idx);
    int local_idx = idx - start_idx;
    refnum[local_idx] = val;

    revive_mismatches(id, idx, val);
  }

  inline __device__ void alloc_msg(message& msg, int idx, size_t size) {
    msg.dst_pe = loc_map[idx];
    msg.env = create_envelope(msgtype::user, size, msg.offset);
  }
  inline __device__ void free_msg(message& msg) { /*TODO*/ }
};

} // namespace charm

#endif // _CHARE_H_
