#ifndef _COMMON_H_
#define _COMMON_H_

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

// Indices into shared memory
enum s_idx : int {
  dst = 0,
  src = 1,
  size = 2,
  env = 3,
  offset = 4,
  msg_size = 5,
  my_pe = 8,
  my_pe_node = 9,
  my_pe_nvshmem = 10
};

#endif // _COMMON_H_
