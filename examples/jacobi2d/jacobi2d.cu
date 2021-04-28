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

__device__ charm::chare_proxy<Block>* block_proxy;

__device__ void charm::register_chares() {
  block_proxy = new charm::chare_proxy<Block>(2);
  block_proxy->add_entry_method(&Block::init);
  block_proxy->add_entry_method(&Block::recv_ghosts);
}

__device__ int device_atoi(const char* str, int strlen) {
  int tmp = 0;
  for (int i = 0; i < strlen; i++) {
    int multiplier = 1;
    for (int j = 0; j < strlen - i - 1; j++) {
      multiplier *= 10;
    }
    tmp += (str[i] - 48) * multiplier;
  }
  return tmp;
}

// Main
__device__ void charm::main(int argc, char** argv, size_t* argvs) {
  // Process command line arguments
  int grid_width = GRID_WIDTH;
  if (argc >= 2) grid_width = device_atoi(argv[1], argvs[1]);
  int grid_height = GRID_HEIGHT;
  if (argc >= 3) grid_height = device_atoi(argv[2], argvs[2]);
  int n_chares = charm::n_pes();
  if (argc >= 4) n_chares = device_atoi(argv[3], argvs[3]);
  int n_iters = N_ITERS;
  if (argc >= 5) n_iters = device_atoi(argv[4], argvs[4]);

  // Set up 2D grid of chares (as square as possible)
  double area[2], surf, bestsurf;
  int ipx, ipy;
  int n_chares_x, n_chares_y;
  area[0] = grid_height;
  area[1] = grid_width;
  bestsurf = 2.0 * (area[0] + area[1]);
  ipx = 1;
  while (ipx <= n_chares) {
    if (n_chares % ipx == 0) {
      ipy = n_chares / ipx;
      surf = 2.0 * (area[0] / ipx + area[1] / ipy);

      if (surf < bestsurf) {
        bestsurf = surf;
        n_chares_x = ipx;
        n_chares_y = ipy;
      }
    }
    ipx++;
  }
  if (n_chares_x * n_chares_y != n_chares) {
    printf("Decomposition failed! %d chares into %d x %d chares\n",
           n_chares, n_chares_x, n_chares_y);
    charm::end();
  }

  int block_width = grid_width / n_chares_x;
  int block_height = grid_height / n_chares_y;

  printf("Grid size: %d x %d\n", grid_width, grid_height);
  printf("Block size: %d x %d\n", block_width, block_height);
  printf("Chare array: %d x %d (%d total)\n", n_chares_x, n_chares_y, n_chares);
  printf("Total number of iterations: %d\n", n_iters);

  Block block;
  block_proxy->create(block, n_chares);
  constexpr int n_params = 5;
  int params[n_params] = { block_width, block_height, n_iters, n_chares_x, n_chares_y };
  for (int i = 0; i < n_chares; i++) {
    block_proxy->invoke(i, 0, params, sizeof(int) * n_params);
  }
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
__device__ void Block::init(void* arg) {
  int* params = (int*)arg;
  int param_idx = 0;

  // Block size and iteration count
  block_width = params[param_idx++];
  block_height = params[param_idx++];
  block_size = (unsigned long long)(block_width+2) * (unsigned long long)(block_height+2);
  n_iters = params[param_idx++];
  iter = 0;

  // Figure out this block's index and its neighbors
  int n_chares_x = params[param_idx++];
  int n_chares_y = params[param_idx++];
  int index = charm::chare::i;
  row = (index / n_chares_y);
  col = (index % n_chares_y);
  neighbor_index[LEFT] = (col == 0) ? -1 : (index-1);
  neighbor_index[RIGHT] = (col == n_chares_y-1) ? -1 : (index+1);
  neighbor_index[TOP] = (row == 0) ? -1 : (index-n_chares_y);
  neighbor_index[BOTTOM] = (row == n_chares_x-1) ? -1 : (index+n_chares_y);
  neighbor_count = 4;
  if (col == 0) neighbor_count--;
  if (col == n_chares_y-1) neighbor_count--;
  if (row == 0) neighbor_count--;
  if (row == n_chares_x-1) neighbor_count--;
  recv_count = 0;

  /*
  printf("%d: I'm (%d,%d) with %d neighbors\n", index, row, col, neighbor_count);
  printf("LEFT: %d, RIGHT: %d, TOP: %d, BOTTOM: %d\n", neighbor_index[LEFT], neighbor_index[RIGHT], neighbor_index[TOP], neighbor_index[BOTTOM]);
  */

  temperature = new DataType[block_size];
  new_temperature = new DataType[block_size];
  ghost_sizes[LEFT] = sizeof(DataType) * block_height;
  ghost_sizes[RIGHT] = sizeof(DataType) * block_height;
  ghost_sizes[TOP] = sizeof(DataType) * block_width;
  ghost_sizes[BOTTOM] = sizeof(DataType) * block_width;
  for (int i = 0; i < N_NEIGHBORS; i++) {
    boundaries[i] = new DataType[ghost_sizes[i] / sizeof(DataType) + 1]; // Need int for direction
    *(int*)boundaries[i] = (i % 2 == 0) ? (i+1) : (i-1);
  }

  dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
  dim3 grid_dim((block_height+2 + block_dim.x - 1) / block_dim.x, (block_width+2 + block_dim.y - 1) / block_dim.y);
  init_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width, block_height);
  cudaDeviceSynchronize();

  start_tp = cuda::std::chrono::system_clock::now();
  send_boundaries();
}

__device__ void Block::send_boundaries() {
  // Pack
  dim3 block_dim(BLOCK_DIM * BLOCK_DIM);
  dim3 grid_dim((block_height + block_dim.x - 1) / block_dim.x);
  if (neighbor_index[LEFT] != -1) {
    DataType* boundary = boundaries[LEFT]+1;
    pack_left_kernel<<<grid_dim, block_dim>>>(temperature, boundary, block_width, block_height);
  }
  if (neighbor_index[RIGHT] != -1) {
    DataType* boundary = boundaries[RIGHT]+1;
    pack_right_kernel<<<grid_dim, block_dim>>>(temperature, boundary, block_width, block_height);
  }
  cudaDeviceSynchronize();

  // Send to neighbors
  if (neighbor_index[LEFT] != -1) {
    block_proxy->invoke(neighbor_index[LEFT], 1, boundaries[LEFT],
                        sizeof(DataType) + ghost_sizes[LEFT]);
  }
  if (neighbor_index[RIGHT] != -1) {
    block_proxy->invoke(neighbor_index[RIGHT], 1, boundaries[RIGHT],
                        sizeof(DataType) + ghost_sizes[RIGHT]);
  }
  if (neighbor_index[TOP] != -1) {
    memcpy(boundaries[TOP]+1,
           temperature + (block_width+2) + 1, ghost_sizes[TOP]);
    block_proxy->invoke(neighbor_index[TOP], 1, boundaries[TOP],
                        sizeof(DataType) + ghost_sizes[TOP]);
  }
  if (neighbor_index[BOTTOM] != -1) {
    memcpy(boundaries[BOTTOM]+1,
           temperature + (block_width+2) * block_height + 1, ghost_sizes[BOTTOM]);
    block_proxy->invoke(neighbor_index[BOTTOM], 1, boundaries[BOTTOM],
                        sizeof(DataType) + ghost_sizes[BOTTOM]);
  }
}

__device__ void Block::recv_ghosts(void* arg) {
  int dir = *(int*)arg;
  DataType* ghost = (DataType*)arg + 1;
  dim3 block_dim(BLOCK_DIM * BLOCK_DIM);
  dim3 grid_dim((block_height + (block_dim.x-1)) / block_dim.x);

  switch (dir) {
    case LEFT:
      unpack_left_kernel<<<grid_dim, block_dim>>>(temperature, ghost, block_width, block_height);
      cudaDeviceSynchronize();
      break;
    case RIGHT:
      unpack_right_kernel<<<grid_dim, block_dim>>>(temperature, ghost, block_width, block_height);
      cudaDeviceSynchronize();
      break;
    case TOP:
      memcpy(temperature + 1, ghost, ghost_sizes[TOP]);
      break;
    case BOTTOM:
      memcpy(temperature + (block_width+2) * (block_height+1) + 1, ghost, ghost_sizes[BOTTOM]);
      break;
    default:
      printf("Wrong direction!\n");
      break;
  }

  if (++recv_count == neighbor_count) {
    recv_count = 0;
    update();
  }
}

__device__ void Block::update() {
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
    printf("Chare (%d,%d) completed %d iterations in %.6lf seconds\n", row, col, iter, diff.count());

    charm::end();
  } else {
    DataType* tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;

    send_boundaries();
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
