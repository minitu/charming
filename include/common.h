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

// Max number of 64-bit values in shared memory
// Used to coordinate fork-join model in scheduler
#define SMEM_CNT_MAX 128

extern __constant__ int c_n_sms;
extern __constant__ int c_n_clusters_dev;
extern __constant__ int c_n_pes_cluster;
extern __constant__ int c_n_ces_cluster;

// Indices into shared memory
enum s_idx : int {
  dst = 0,
  src = 1,
  size = 2,
  env = 3,
  offset = 4,
  my_pe = 8,
  my_pe_node = 9,
  my_pe_nvshmem = 10
};

__device__ __forceinline__ int get_local_rank_from_pe(int pe) {
  int local_pe = pe % (c_n_clusters_dev * c_n_pes_cluster);
  int cluster = local_pe / c_n_pes_cluster;
  return (cluster * (c_n_pes_cluster + c_n_ces_cluster) + local_pe % c_n_pes_cluster);
}

__device__ __forceinline__ int get_local_rank_from_ce(int ce) {
  int local_ce = ce % (c_n_clusters_dev * c_n_ces_cluster);
  int cluster = local_ce / c_n_ces_cluster;
  return (cluster * (c_n_pes_cluster + c_n_ces_cluster) + c_n_pes_cluster + local_ce % c_n_ces_cluster);
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
  return (pe / c_n_pes_cluster);
}

#endif // _COMMON_H_
