#ifndef _COMMON_H_
#define _COMMON_H_

#include <nvfunctional> // For placement new

// Print functions
#define PINFO(...) printf("[INFO] " __VA_ARGS__)
#define PERROR(...) printf("[ERROR] " __VA_ARGS__)
#ifdef DEBUG
#define PDEBUG(...) printf("[DEBUG] " __VA_ARGS__)
#else
#define PDEBUG(...) do {} while (0)
#endif

extern __constant__ int c_n_sms;
extern __constant__ int c_my_dev;
extern __constant__ int c_my_dev_node;
extern __constant__ int c_n_devs;
extern __constant__ int c_n_devs_node;
extern __constant__ int c_n_nodes;
extern __constant__ int c_n_pes;
extern __constant__ int c_n_pes_node;

#ifdef SM_LEVEL
// Max number of 64-bit values in shared memory
// Used to coordinate fork-join model in scheduler
#define SMEM_CNT_MAX 128

extern __constant__ int c_n_clusters_dev;
extern __constant__ int c_cluster_size;
extern __constant__ int c_n_pes_cluster;
extern __constant__ int c_n_ces_cluster;
extern __constant__ int c_n_ces;
extern __constant__ int c_n_ces_node;

// Indices into shared memory
namespace s_idx {
  enum : int {
    dst = 0,
    src = 1,
    size = 2,
    env = 3,
    offset = 4,
    local_rank = 5,
    dst_ce = 6,
    dev = 7,
    is_pe = 8,
    my_pe = 9,
    my_ce = 10
  };
}

__device__ __forceinline__ int get_local_rank_from_pe(int pe) {
  int pe_dev = pe % (c_n_clusters_dev * c_n_pes_cluster);
  int cluster_dev = pe_dev / c_n_pes_cluster;
  return (cluster_dev * c_cluster_size + pe_dev % c_n_pes_cluster);
}

__device__ __forceinline__ int get_local_rank_from_ce(int ce) {
  int ce_dev = ce % (c_n_clusters_dev * c_n_ces_cluster);
  int cluster_dev = ce_dev / c_n_ces_cluster;
  return (cluster_dev * c_cluster_size + c_n_pes_cluster + ce_dev % c_n_ces_cluster);
}

__device__ __forceinline__ int get_ce_in_dev(int ce) {
  return ce % (c_n_clusters_dev * c_n_ces_cluster);
}

__device__ __forceinline__ int get_dev_from_pe(int pe) {
  // Divide PE number by number of PEs per device
  return (pe / (c_n_clusters_dev * c_n_pes_cluster));
}

__device__ __forceinline__ int get_dev_from_ce(int ce) {
  // Divide CE number by number of CEs per device
  return (ce / (c_n_clusters_dev * c_n_ces_cluster));
}

__device__ __forceinline__ int get_ce_from_pe(int pe) {
  // Round-robin assignment of PEs to CEs
  int cluster_global = pe / c_n_pes_cluster;
  int start_ce = cluster_global * c_n_ces_cluster;
  int rank_in_cluster = get_local_rank_from_pe(pe) % c_cluster_size;
  return (start_ce + (rank_in_cluster % c_n_ces_cluster));
}
#else
#define GID (blockDim.x * blockIdx.x + threadIdx.x)
#define BID (blockIdx.x)
#endif // SM_LEVEL

#endif // _COMMON_H_
