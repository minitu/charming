#ifndef _JACOBI2D_H_
#define _JACOBI2D_H_

#include <cuda/std/chrono>
#include <charming.h>

#define ALIGN_SIZE 16
#define N_NEIGHBORS 4

typedef float real;
typedef cuda::std::chrono::time_point<cuda::std::chrono::system_clock> cuda_tp;
typedef cuda::std::chrono::duration<double> cuda_dur;

struct Block : charm::chare {
  int nx;
  int ny;
  int iter;
  int iter_max;
  int npes;
  int mype;
  int ngpus;
  int npes_per_gpu;
  int mype_local;
  int recv_count;
  int term_count;
#ifndef SM_LEVEL
  bool done;
  bool end;
#endif

  real* a;
  real* a_new;
  size_t a_count;
  size_t a_size;
  size_t a_new_size;
  uint64_t* sync_arr;
  size_t sync_count;
  size_t sync_size;
  long synccounter;

  int chunk_size;

  int iy_start_global;
  int iy_end_global;
  int iy_start;
  int iy_end;
  int top_pe;
  int bottom_pe;
  int iy_end_top;
  int iy_start_bottom;

  cuda_tp start_tp;
  cuda_tp end_tp;

  __device__ Block() {}
  __device__ void init(void* arg);
  __device__ void iterate();
  __device__ void begin();
  __device__ void resume(void* arg);
  //__device__ void update();
  //__device__ void send_halo();
  //__device__ void recv_halo(void* arg);
  //__device__ void terminate(void* arg);
};

// Entry methods
__device__ void entry_init(Block& c, void* arg) { c.init(arg); }
__device__ void entry_resume(Block& c, void* arg) { c.resume(arg); }
/*
__device__ void entry_recv_halo(Block& c, void* arg) { c.recv_halo(arg); }
__device__ void entry_terminate(Block& c, void* arg) { c.terminate(arg); }
*/

#endif
