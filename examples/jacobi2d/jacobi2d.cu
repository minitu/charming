#include <stdio.h>
#include "jacobi2d.h"

#define LEFT 0
#define RIGHT 1
#define TOP 2
#define BOTTOM 3

#define BLOCK_WIDTH 65536
#define BLOCK_HEIGHT 65536
#define N_ITERS 1000
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
  int block_width = BLOCK_WIDTH;
  if (argc >= 2) block_width = device_atoi(argv[1], argvs[1]);
  int block_height = BLOCK_HEIGHT;
  if (argc >= 3) block_height = device_atoi(argv[2], argvs[2]);
  int n_iters = N_ITERS;
  if (argc >= 4) n_iters = device_atoi(argv[3], argvs[3]);

  printf("Setting block size: %d x %d\n", block_width, block_height);
  printf("Setting total number of iterations: %d\n", n_iters);

  int n_chares = charm::n_pes();
  int sqrt_n = (int)sqrt((float)n_chares);
  if (sqrt_n * sqrt_n != n_chares) {
    printf("Number of chares (PEs) %d should be a square number!\n", n_chares);
    return;
  }

  Block block;
  block_proxy->create(block, n_chares);
  constexpr int n_params = 4;
  int params[n_params] = { block_width, block_height, n_iters, sqrt_n };
  for (int i = 0; i < charm::n_pes(); i++) {
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
  sqrt_n = params[param_idx++];
  int index = charm::chare::i;
  row = (index / sqrt_n);
  col = (index % sqrt_n);
  neighbor_index[LEFT] = (col == 0) ? -1 : (index-1);
  neighbor_index[RIGHT] = (col == sqrt_n-1) ? -1 : (index+1);
  neighbor_index[TOP] = (row == 0) ? -1 : (index-sqrt_n);
  neighbor_index[BOTTOM] = (row == sqrt_n-1) ? -1 : (index+sqrt_n);
  neighbor_count = 4;
  if (col == 0) neighbor_count--;
  if (col == sqrt_n-1) neighbor_count--;
  if (row == 0) neighbor_count--;
  if (row == sqrt_n-1) neighbor_count--;
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
    ghosts[i] = new DataType[ghost_sizes[i] / sizeof(DataType) + 1]; // Need int for direction
    *(int*)ghosts[i] = (i % 2 == 0) ? (i+1) : (i-1);
  }

  dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
  dim3 grid_dim((block_height+2 + block_dim.x - 1) / block_dim.x, (block_width+2 + block_dim.y - 1) / block_dim.y);
  init_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width, block_height);
  cudaDeviceSynchronize();

  start_tp = cuda::std::chrono::system_clock::now();
  send_ghosts();
}

__device__ void Block::send_ghosts() {
  // Pack
  dim3 block_dim(BLOCK_DIM * BLOCK_DIM);
  dim3 grid_dim((block_height + block_dim.x - 1) / block_dim.x);
  if (neighbor_index[LEFT] != -1) {
    DataType* ghost = ghosts[LEFT]+1;
    pack_left_kernel<<<grid_dim, block_dim>>>(temperature, ghost, block_width, block_height);
  }
  if (neighbor_index[RIGHT] != -1) {
    DataType* ghost = ghosts[RIGHT]+1;
    pack_right_kernel<<<grid_dim, block_dim>>>(temperature, ghost, block_width, block_height);
  }
  cudaDeviceSynchronize();

  // Send to neighbors
  if (neighbor_index[LEFT] != -1) {
    block_proxy->invoke(neighbor_index[LEFT], 1, ghosts[LEFT],
                        sizeof(DataType) + ghost_sizes[LEFT]);
  }
  if (neighbor_index[RIGHT] != -1) {
    block_proxy->invoke(neighbor_index[RIGHT], 1, ghosts[RIGHT],
                        sizeof(DataType) + ghost_sizes[RIGHT]);
  }
  if (neighbor_index[TOP] != -1) {
    memcpy(ghosts[TOP]+1,
           temperature + (block_width+2) + 1, ghost_sizes[TOP]);
    block_proxy->invoke(neighbor_index[TOP], 1, ghosts[TOP],
                        sizeof(DataType) + ghost_sizes[TOP]);
  }
  if (neighbor_index[BOTTOM] != -1) {
    memcpy(ghosts[BOTTOM]+1,
           temperature + (block_width+2) * block_height + 1, ghost_sizes[BOTTOM]);
    block_proxy->invoke(neighbor_index[BOTTOM], 1, ghosts[BOTTOM],
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

    charm::end(charm::my_pe());
  } else {
    DataType* tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;

    send_ghosts();
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
