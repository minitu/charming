#include <stdio.h>
#include "jacobi2d.h"

#define LEFT 0
#define RIGHT 1
#define TOP 2
#define BOTTOM 3

#define GRID_WIDTH 65536
#define GRID_HEIGHT 65536
#define N_ITERS 100
#define BLOCK_DIM 16

#define IDX(i,j) ((block_width+2)*(i) + (j))

// Main
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, n_procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

  // Process command line arguments
  int grid_width = GRID_WIDTH;
  if (argc >= 2) grid_width = atoi(argv[1]);
  int grid_height = GRID_HEIGHT;
  if (argc >= 3) grid_height = atoi(argv[2]);
  int n_iters = N_ITERS;
  if (argc >= 4) n_iters = atoi(argv[3]);

  // Set up 2D grid of processes (as square as possible)
  double area[2], surf, bestsurf;
  int ipx, ipy;
  int n_procs_x, n_procs_y;
  area[0] = grid_height;
  area[1] = grid_width;
  bestsurf = 2.0 * (area[0] + area[1]);
  ipx = 1;
  while (ipx <= n_procs) {
    if (n_procs % ipx == 0) {
      ipy = n_procs / ipx;
      surf = 2.0 * (area[0] / ipx + area[1] / ipy);

      if (surf < bestsurf) {
        bestsurf = surf;
        n_procs_x = ipx;
        n_procs_y = ipy;
      }
    }
    ipx++;
  }
  if (n_procs_x * n_procs_y != n_procs) {
    printf("Decomposition failed! %d procs into %d x %d procs\n",
           n_procs, n_procs_x, n_procs_y);
    exit(-1);
  }

  int block_width = grid_width / n_procs_x;
  int block_height = grid_height / n_procs_y;

  if (rank == 0) {
    printf("Grid size: %d x %d\n", grid_width, grid_height);
    printf("Block size: %d x %d\n", block_width, block_height);
    printf("Proc array: %d x %d (%d total)\n", n_procs_x, n_procs_y, n_procs);
    printf("Total number of iterations: %d\n", n_iters);
  }

  constexpr int n_params = 5;
  int params[n_params] = { block_width, block_height, n_iters, n_procs_x, n_procs_y };
  Block block;
  block.init(params);
  for (int i = 0; i < n_iters; i++) {
    block.send_boundaries();
    block.recv_ghosts();
    block.update();
  }

  MPI_Finalize();

  return 0;
}

__global__ void init_kernel(DataType* temperature, DataType* new_temperature,
                            int block_width, int block_height);
__global__ void pack_left_kernel(DataType* temperature, DataType* ghost,
                                 int block_width, int block_height);
__global__ void pack_right_kernel(DataType* temperature, DataType* ghost,
                                  int block_width, int block_height);
__global__ void unpack_left_kernel(DataType* temperature, DataType* ghost,
                                   int block_width, int block_height);
__global__ void unpack_right_kernel(DataType* temperature, DataType* ghost,
                                    int block_width, int block_height);
__global__ void jacobi_kernel(DataType* temperature, DataType* new_temperature,
                              int block_width, int block_height);

// Entry methods
void Block::init(void* arg) {
  int* params = (int*)arg;
  int param_idx = 0;

  // Block size and iteration count
  block_width = params[param_idx++];
  block_height = params[param_idx++];
  block_size = (unsigned long long)(block_width+2) * (unsigned long long)(block_height+2);
  n_iters = params[param_idx++];
  iter = 0;

  // Figure out this block's index and its neighbors
  int n_procs_x = params[param_idx++];
  int n_procs_y = params[param_idx++];
  MPI_Comm_rank(MPI_COMM_WORLD, &index);
  row = (index / n_procs_y);
  col = (index % n_procs_y);
  neighbor_index[LEFT] = (col == 0) ? MPI_PROC_NULL : (index-1);
  neighbor_index[RIGHT] = (col == n_procs_y-1) ? MPI_PROC_NULL : (index+1);
  neighbor_index[TOP] = (row == 0) ? MPI_PROC_NULL : (index-n_procs_y);
  neighbor_index[BOTTOM] = (row == n_procs_x-1) ? MPI_PROC_NULL : (index+n_procs_y);
  neighbor_count = N_NEIGHBORS;
  if (col == 0) neighbor_count--;
  if (col == n_procs_y-1) neighbor_count--;
  if (row == 0) neighbor_count--;
  if (row == n_procs_x-1) neighbor_count--;

  cudaMalloc(&temperature, sizeof(DataType) * block_size);
  cudaMalloc(&new_temperature, sizeof(DataType) * block_size);
  ghost_sizes[LEFT] = sizeof(DataType) * block_height;
  ghost_sizes[RIGHT] = sizeof(DataType) * block_height;
  ghost_sizes[TOP] = sizeof(DataType) * block_width;
  ghost_sizes[BOTTOM] = sizeof(DataType) * block_width;
  for (int i = 0; i < N_NEIGHBORS; i++) {
    cudaMalloc(&boundaries[i], ghost_sizes[i]);
    cudaMalloc(&ghosts[i], ghost_sizes[i]);
  }

  dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
  dim3 grid_dim((block_height+2 + block_dim.x - 1) / block_dim.x, (block_width+2 + block_dim.y - 1) / block_dim.y);
  init_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width, block_height);
  cudaDeviceSynchronize();

  start_tp = cuda::std::chrono::system_clock::now();
}

void Block::send_boundaries() {
  // Pack
  dim3 block_dim(BLOCK_DIM * BLOCK_DIM);
  dim3 grid_dim((block_height + block_dim.x - 1) / block_dim.x);
  if (neighbor_index[LEFT] != MPI_PROC_NULL) {
    pack_left_kernel<<<grid_dim, block_dim>>>(temperature, boundaries[LEFT], block_width, block_height);
  }
  if (neighbor_index[RIGHT] != MPI_PROC_NULL) {
    pack_right_kernel<<<grid_dim, block_dim>>>(temperature, boundaries[RIGHT], block_width, block_height);
  }
  if (neighbor_index[TOP] != MPI_PROC_NULL) {
    cudaMemcpy(boundaries[TOP], temperature + (block_width+2) + 1, ghost_sizes[TOP],
               cudaMemcpyDeviceToDevice);
  }
  if (neighbor_index[BOTTOM] != MPI_PROC_NULL) {
    cudaMemcpy(boundaries[BOTTOM], temperature + (block_width+2) * block_height + 1,
              ghost_sizes[BOTTOM], cudaMemcpyDeviceToDevice);
  }
  cudaDeviceSynchronize();

  // Send to neighbors
  MPI_Isend(boundaries[LEFT], ghost_sizes[LEFT], MPI_CHAR, neighbor_index[LEFT],
            iter * N_NEIGHBORS + RIGHT, MPI_COMM_WORLD, &requests[LEFT]);
  MPI_Isend(boundaries[RIGHT], ghost_sizes[RIGHT], MPI_CHAR, neighbor_index[RIGHT],
            iter * N_NEIGHBORS + LEFT, MPI_COMM_WORLD, &requests[RIGHT]);
  MPI_Isend(boundaries[TOP], ghost_sizes[TOP], MPI_CHAR, neighbor_index[TOP],
            iter * N_NEIGHBORS + BOTTOM, MPI_COMM_WORLD, &requests[TOP]);
  MPI_Isend(boundaries[BOTTOM], ghost_sizes[BOTTOM], MPI_CHAR, neighbor_index[BOTTOM],
            iter * N_NEIGHBORS + TOP, MPI_COMM_WORLD, &requests[BOTTOM]);
}

void Block::recv_ghosts() {
  // Receive from neighbors
  MPI_Irecv(ghosts[LEFT], ghost_sizes[LEFT], MPI_CHAR, neighbor_index[LEFT],
            iter * N_NEIGHBORS + LEFT, MPI_COMM_WORLD, &requests[N_NEIGHBORS + LEFT]);
  MPI_Irecv(ghosts[RIGHT], ghost_sizes[RIGHT], MPI_CHAR, neighbor_index[RIGHT],
            iter * N_NEIGHBORS + RIGHT, MPI_COMM_WORLD, &requests[N_NEIGHBORS + RIGHT]);
  MPI_Irecv(ghosts[TOP], ghost_sizes[TOP], MPI_CHAR, neighbor_index[TOP],
            iter * N_NEIGHBORS + TOP, MPI_COMM_WORLD, &requests[N_NEIGHBORS + TOP]);
  MPI_Irecv(ghosts[BOTTOM], ghost_sizes[BOTTOM], MPI_CHAR, neighbor_index[BOTTOM],
            iter * N_NEIGHBORS + BOTTOM, MPI_COMM_WORLD, &requests[N_NEIGHBORS + BOTTOM]);

  for (int i = 0; i < N_NEIGHBORS * 2; i++) {
    MPI_Waitall(N_NEIGHBORS * 2, requests, statuses);
  }

  dim3 block_dim(BLOCK_DIM * BLOCK_DIM);
  dim3 grid_dim((block_height + (block_dim.x-1)) / block_dim.x);

  // Unpack
  if (neighbor_index[LEFT] != MPI_PROC_NULL) {
      unpack_left_kernel<<<grid_dim, block_dim>>>(temperature, ghosts[LEFT], block_width, block_height);
  }
  if (neighbor_index[RIGHT] != MPI_PROC_NULL) {
      unpack_right_kernel<<<grid_dim, block_dim>>>(temperature, ghosts[RIGHT], block_width, block_height);
  }
  if (neighbor_index[TOP] != MPI_PROC_NULL) {
      cudaMemcpy(temperature + 1, ghosts[TOP], ghost_sizes[TOP], cudaMemcpyDeviceToDevice);
  }
  if (neighbor_index[BOTTOM] != MPI_PROC_NULL) {
      cudaMemcpy(temperature + (block_width+2) * (block_height+1) + 1, ghosts[BOTTOM],
                 ghost_sizes[BOTTOM], cudaMemcpyDeviceToDevice);
  }
  cudaDeviceSynchronize();
}

void Block::update() {
  dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
  dim3 grid_dim((block_height + (block_dim.x-1)) / block_dim.x, (block_width + (block_dim.y-1)) / block_dim.y);
  jacobi_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width, block_height);
  cudaDeviceSynchronize();

  /*
  printf("OLD\n");
  for (int i = 0; i < block_height+2; i++) {
    for (int j = 0; j < block_width+2; j++) {
      printf("%-10.3lf ", temperature[IDX(i,j)]);
    }
    printf("\n");
  }

  printf("NEW\n");
  for (int i = 0; i < block_height+2; i++) {
    for (int j = 0; j < block_width+2; j++) {
      printf("%-10.3lf ", new_temperature[IDX(i,j)]);
    }
    printf("\n");
  }
  */

  if (++iter == n_iters) {
    end_tp = cuda::std::chrono::system_clock::now();
    cuda::std::chrono::duration<double> diff = end_tp - start_tp;
    double total_time = diff.count();
    printf("MPI rank (%d,%d) completed %d iterations in %.6lf seconds\n", row, col, iter, diff.count());
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (index == 0) {
      printf("Max time: %.6lf seconds\n", max_time);
    }
  } else {
    DataType* tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;
  }
}

// GPU kernels
__global__ void init_kernel(DataType* temperature, DataType* new_temperature,
                            int block_width, int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i > 0 && i < block_height+1 && j > 0 && j < block_width+1) {
    temperature[IDX(i,j)] = 0;
    new_temperature[IDX(i,j)] = 0;
  } else if (i < block_height+2 && j < block_width+2) {
    temperature[IDX(i,j)] = 10;
    new_temperature[IDX(i,j)] = 10;
  }
}

__global__ void pack_left_kernel(DataType* temperature, DataType* ghost,
                                 int block_width, int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  if (i < block_height+1) {
    ghost[i-1] = temperature[IDX(i,1)];
  }
}

__global__ void pack_right_kernel(DataType* temperature, DataType* ghost,
                                  int block_width, int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  if (i < block_height+1) {
    ghost[i-1] = temperature[IDX(i,block_width)];
  }
}

__global__ void unpack_left_kernel(DataType* temperature, DataType* ghost,
                                   int block_width, int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  if (i < block_height+1) {
    temperature[IDX(i,0)] = ghost[i-1];
  }
}

__global__ void unpack_right_kernel(DataType* temperature, DataType* ghost,
                                    int block_width, int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  if (i < block_height+1) {
    temperature[IDX(i,block_width+1)] = ghost[i-1];
  }
}

__global__ void jacobi_kernel(DataType* temperature, DataType* new_temperature,
                              int block_width, int block_height) {
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
  if (i < block_height+1 && j < block_width+1) {
    new_temperature[IDX(i,j)] = (temperature[IDX(i,j)] + temperature[IDX(i,j-1)]
      + temperature[IDX(i,j+1)] + temperature[IDX(i-1,j)] + temperature[IDX(i+1,j)]) * 0.2;
  }
}
