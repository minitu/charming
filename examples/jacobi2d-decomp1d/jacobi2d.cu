#include <stdio.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "jacobi2d.h"

#define PI 3.141592

#define GRID_WIDTH 16384
#define GRID_HEIGHT 16384
#define N_ITERS 1000

#ifdef SM_LEVEL
// SM-level scheduling
#define GID (threadIdx.x)
#define GROUP_SIZE (blockDim.x)
#define BARRIER_LOCAL __syncthreads()
__shared__ charm::chare_proxy<Block>* block_proxy;
#else
// GPU-level scheduling
#define GID (blockDim.x * blockIdx.x + threadIdx.x)
#define GROUP_SIZE (gridDim.x * blockDim.x)
#define BARRIER_LOCAL charm::barrier_local()
__device__ charm::chare_proxy<Block>* block_proxy;
__device__ int* params;
__managed__ real* a_global;
__managed__ real* a_new_global;
__managed__ uint64_t* sync_arr_global;
#endif

void charm::main_host(int argc, char** argv) {
  assert(argc == 5);
  int n_chares = atoi(argv[1]);
  int nx = atoi(argv[2]);
  int ny = atoi(argv[3]);
  int iter_max = atoi(argv[4]);
  int pe = nvshmem_my_pe();
  if (pe == 0) {
    printf("Jacobi2D: Chares %d, nx %d, ny %d, iter_max %d\n",
        n_chares, nx, ny, iter_max);
  }

  // Compute chunk size and allocate memory
  int n_pes = nvshmem_n_pes();
  assert(n_chares % n_pes == 0);
  int n_chares_per_pe = n_chares / n_pes;
  int chunk_size_low = (ny - 2) / n_chares;
  int chunk_size_high = chunk_size_low + 1;
  size_t a_global_size = nx * (chunk_size_high + 2) * sizeof(real) * n_chares_per_pe;
  size_t a_new_global_size = a_global_size;
  size_t sync_global_size = 2 * sizeof(uint64_t) * n_chares_per_pe;

  a_global = (real*)nvshmem_malloc(a_global_size);
  a_new_global = (real*)nvshmem_malloc(a_new_global_size);
  sync_arr_global = (uint64_t*)nvshmem_malloc(sync_global_size);
  assert(a_global && a_new_global && sync_arr_global);
}

__device__ void charm::main(int argc, char** argv, size_t* argvs, int pe) {
  // Execute on all elements

  // Process command line arguments
  int n_chares = charm::n_pes();
  int grid_width = GRID_WIDTH;
  int grid_height = GRID_HEIGHT;
  int n_iters = N_ITERS;
  if (argc >= 2) n_chares = charm::device_atoi(argv[1], argvs[1]);
  if (argc >= 3) grid_width = charm::device_atoi(argv[2], argvs[2]);
  if (argc >= 4) grid_height = charm::device_atoi(argv[3], argvs[3]);
  if (argc >= 5) n_iters = charm::device_atoi(argv[4], argvs[4]);

  if (GID == 0) {
    // Create chare proxy and register entry methods
    block_proxy = new charm::chare_proxy<Block>();
    block_proxy->add_entry_method<&entry_init>();
    /*
    block_proxy->add_entry_method<&entry_recv_halo>();
    block_proxy->add_entry_method<&entry_terminate>();
    */

    // Create chares
    block_proxy->create(n_chares);
  }
  BARRIER_LOCAL;

  barrier();

  // Executed only on PE 0
  if (pe == 0) {
    constexpr int n_params = 3;
#ifdef SM_LEVEL
    __shared__ int params[n_params];
#endif

    if (GID == 0) {
#ifndef SM_LEVEL
      params = new int[n_params];
#endif

      params[0] = grid_width;
      params[1] = grid_height;
      params[2] = n_iters;

      printf("Chares: %d\n", n_chares);
      printf("Grid size: %d x %d\n", grid_width, grid_height);
      printf("Iterations: %d\n", n_iters);
    }
    BARRIER_LOCAL;

    block_proxy->invoke_all(0, params, sizeof(int) * n_params);
  }
}

__device__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                      const real pi, const int offset, const int nx,
                                      const int my_ny, int ny);

__device__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              const int iy_start, const int iy_end, const int nx, const int top_pe,
                              const int top_iy, const int bottom_pe, const int bottom_iy,
                              const size_t a_count, const int npes_per_gpu);

__device__ void syncneighborhood_kernel(int my_pe, int num_pes, uint64_t* sync_arr,
                                        long counter, size_t sync_count, int npes_per_gpu);

// Entry methods
__device__ void Block::init(void* arg) {
  if (GID == 0) {
    int* params = (int*)arg;
    nx = params[0];
    ny = params[1];
    iter_max = params[2];
    npes = charm::chare::n;
    mype = charm::chare::i;
    recv_count = 0;
    term_count = 0;
    printf("Block %3d init on PE %d\n", mype, charm::my_pe());

    // Compute chunk size and allocate memory
    int chunk_size_low = (ny - 2) / npes;
    int chunk_size_high = chunk_size_low + 1;
    int num_ranks_low = npes * chunk_size_low + npes - (ny - 2);
    if (mype < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    ngpus = charm::n_pes();
    npes_per_gpu = npes / ngpus;
    mype_local = mype % npes_per_gpu;
    a_count = nx * (chunk_size_high + 2);
    a_size = a_new_size = a_count * sizeof(real);
    a = a_global + a_count * mype_local;
    a_new = a_new_global + a_count * mype_local;
    assert(a && a_new);

    // Signal array for neighborhood synchronization
    sync_count = 2;
    sync_size = sync_count * sizeof(uint64_t);
    sync_arr = sync_arr_global + sync_count * mype_local;
    synccounter = 1;
    assert(sync_arr);

    // Calculate local domain boundaries
    if (mype < num_ranks_low) {
        iy_start_global = mype * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (mype - num_ranks_low) * chunk_size_high + 1;
    }
    iy_end_global = iy_start_global + chunk_size - 1;
    iy_end_global = min(iy_end_global, ny - 4); // Do not process boundaries

    iy_start = 1;
    iy_end = (iy_end_global - iy_start_global + 1) + iy_start;

    // Calculate boundary indices for top and bottom boundaries
    top_pe = mype > 0 ? mype - 1 : (npes - 1);
    bottom_pe = (mype + 1) % npes;

    iy_end_top = (top_pe < num_ranks_low) ? chunk_size_low + 1 : chunk_size_high + 1;
    iy_start_bottom = 0;

    // Set initial reference number
    //block_proxy->set_refnum(mype, iter);
  }
  BARRIER_LOCAL;

#ifdef SM_LEVEL
  charm::memset_kernel_block(a, 0, a_size);
  charm::memset_kernel_block(a_new, 0, a_new_size);
  charm::memset_kernel_block(sync_arr, 0, sync_size);
#else
  charm::memset_kernel_grid(a, 0, a_size);
  charm::memset_kernel_grid(a_new, 0, a_new_size);
  charm::memset_kernel_grid(sync_arr, 0, sync_size);
#endif

  initialize_boundaries(a_new, a, PI, iy_start_global - 1, nx, chunk_size, ny - 2);

  // Start iteration
  iterate();
}

__device__ void Block::iterate() {
  if (GID == 0) {
    start_tp = cuda::std::chrono::system_clock::now();
  }
  BARRIER_LOCAL;

  for (int iter = 0; iter < iter_max; iter++) {
    // Execute Jacobi update kernel
    jacobi_kernel(a_new, a, iy_start, iy_end, nx, top_pe, iy_end_top,
        bottom_pe, iy_start_bottom, a_count, npes_per_gpu);
    BARRIER_LOCAL;

    // Neighborhood synchronization
    if (GID == 0) {
      syncneighborhood_kernel(mype, npes, sync_arr, synccounter, sync_count, npes_per_gpu);
      synccounter++;
    }
    BARRIER_LOCAL;

    // Swap pointers
    /*
    if (GID == 0) {
      real* temp = a;
      a = a_new;
      a_new = temp;
    }
    BARRIER_LOCAL;
    */
  }

  if (GID == 0) {
    end_tp = cuda::std::chrono::system_clock::now();
    cuda_dur diff = end_tp - start_tp;
    printf("Block %3d completed %4d iterations in %lf s\n", mype, iter_max, diff.count());
  }
  BARRIER_LOCAL;
}

/*
__device__ void Block::update() {
  // Execute Jacobi update kernel
  jacobi_kernel(a_new, a, iy_start, iy_end, nx);

  // Send halo to neighbors
  send_halo();
}

__device__ void Block::send_halo() {
  block_proxy->invoke(top_pe, 1, a_new + iy_start * nx, nx * sizeof(real), iter);
  block_proxy->invoke(bottom_pe, 1, a_new + (iy_end - 1) * nx, nx * sizeof(real), iter);
}

__device__ void Block::recv_halo(void* arg) {
  // TODO: Figure out if halo came from the top or bottom neighbor & memcpy
#ifdef SM_LEVEL
  __shared__ bool done;
  __shared__ bool end;
#endif

  if (GID == 0) {
    done = false;
    end = false;

    if (++recv_count == 2) {
      // Received halos from both neighbors
      printf("Chare %3d completed iteration %d\n", mype, iter);
      recv_count = 0;
      done = true;

      // Set reference number for next iteration
      iter++;
      block_proxy->set_refnum(mype, iter);

      if (iter == iter_max) {
        end = true;
        printf("Chare %3d completed all %d iterations\n", mype, iter_max);
      }
    }
  }
  BARRIER_LOCAL;

  if (done) {
    if (end) {
      block_proxy->invoke(0, 2, nullptr, 0, -1);
    } else {
      update();
    }
  }
}

__device__ void Block::terminate(void* arg) {
  // Terminate only when all chares have finished
  if (GID == 0) {
    term_count++;
  }
  BARRIER_LOCAL;

  if (term_count == npes) {
    charm::end();
  }
}
*/

__device__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                      const real pi, const int offset, const int nx,
                                      const int my_ny, int ny) {
  for (int iy = GID; iy < my_ny; iy += GROUP_SIZE) {
    const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
    a[(iy + 1) * nx + 0] = y0;
    a[(iy + 1) * nx + (nx - 1)] = y0;
    a_new[(iy + 1) * nx + 0] = y0;
    a_new[(iy + 1) * nx + (nx - 1)] = y0;
  }
}

__device__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              const int iy_start, const int iy_end, const int nx, const int top_pe,
                              const int top_iy, const int bottom_pe, const int bottom_iy,
                              const size_t a_count, const int npes_per_gpu) {
  for (int iy = iy_start; iy < iy_end; iy++) {
    for (int ix = GID + 1; ix < (nx - 1); ix += GROUP_SIZE) {
      const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                   a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
      a_new[iy * nx + ix] = new_val;
    }
  }

  BARRIER_LOCAL;

  /* Communicate the boundaries */
  int top_gpu = top_pe / npes_per_gpu;
  int bottom_gpu = bottom_pe / npes_per_gpu;
  int top_pe_local = top_pe % npes_per_gpu;
  int bottom_pe_local = bottom_pe % npes_per_gpu;
  real* a_new_top = a_new_global + a_count * top_pe_local;
  real* a_new_bottom = a_new_global + a_count * bottom_pe_local;
  for (int block_ix = blockIdx.x * blockDim.x + 1; block_ix < ((nx + blockDim.x - 1) / blockDim.x);
      block_ix += gridDim.x) {
    nvshmemx_float_put_nbi_block(a_new_top + top_iy * nx + block_ix, a_new + iy_start * nx + block_ix,
                                 min(blockDim.x, nx - 1 - block_ix), top_gpu);
    nvshmemx_float_put_nbi_block(a_new_bottom + bottom_iy * nx + block_ix,
                                 a_new + (iy_end - 1) * nx + block_ix,
                                 min(blockDim.x, nx - 1 - block_ix), bottom_gpu);
  }
}

__device__ void syncneighborhood_kernel(int my_pe, int num_pes, uint64_t* sync_arr,
                                        long counter, size_t sync_count, int npes_per_gpu) {
    int next_rank = (my_pe + 1) % num_pes;
    int prev_rank = (my_pe == 0) ? num_pes - 1 : my_pe - 1;
    nvshmem_quiet(); /* To ensure all prior nvshmem operations have been completed */

    /* Notify neighbors about arrival */
    int next_gpu = next_rank / npes_per_gpu;
    int prev_gpu = prev_rank / npes_per_gpu;
    int next_rank_local = next_rank % npes_per_gpu;
    int prev_rank_local = prev_rank % npes_per_gpu;
    uint64_t* sync_arr_next = sync_arr_global + sync_count * next_rank_local;
    uint64_t* sync_arr_prev = sync_arr_global + sync_count * prev_rank_local;
    /*
    printf("Block %d signaling %d (GPU %d, %p) and %d (GPU %d, %p)\n", my_pe,
        next_rank, next_gpu, sync_arr_next, prev_rank, prev_gpu, sync_arr_prev + 1);
        */
    nvshmemx_signal_op(sync_arr_next, counter, NVSHMEM_SIGNAL_SET, next_gpu);
    nvshmemx_signal_op(sync_arr_prev + 1, counter, NVSHMEM_SIGNAL_SET, prev_gpu);

    /* Wait for neighbors notification */
    //printf("Block %d waiting on signals at %p and %p\n", my_pe, sync_arr, sync_arr + 1);
    nvshmem_uint64_wait_until_all(sync_arr, 2, NULL, NVSHMEM_CMP_GE, counter);
}
