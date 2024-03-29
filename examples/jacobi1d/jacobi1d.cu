#include <stdio.h>
#include "jacobi1d.h"

#define LEFT 0
#define RIGHT 1

#define BLOCK_WIDTH 128000000
#define N_ITERS 1000
#define BLOCK_DIM 256

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
  if (argc >= 2) {
    block_width = device_atoi(argv[1], argvs[1]);
  }
  int n_iters = N_ITERS;
  if (argc >= 3) {
    n_iters = device_atoi(argv[2], argvs[2]);
  }
  printf("Setting block width: %d\n", block_width);
  printf("Setting total number of iterations: %d\n", n_iters);

  Block block;

  block_proxy->create(block, charm::n_pes());
  int params[2] = { block_width, n_iters };
  for (int i = 0; i < charm::n_pes(); i++) {
    block_proxy->invoke(i, 0, params, sizeof(int) * 2);
  }
}

__global__ void init_kernel(DataType* temperature, DataType* new_temperature,
                            int block_width);
__global__ void boundary_kernel(DataType* temperature, DataType* new_temperature,
                                int block_width);
__global__ void jacobi_kernel(DataType* temperature, DataType* new_temperature,
                              int block_width);

// Entry methods
__device__ void Block::init(void* arg) {
  int* params = (int*)arg;
  index = charm::chare::i;
  block_width = params[0];
  iter = 0;
  n_iters = params[1];
  data_size = block_width + GHOST_SIZE*2;
  temperature = new DataType[data_size];
  new_temperature = new DataType[data_size];
  left_index = (index == 0) ? -1 : (index-1);
  right_index = (index == charm::n_pes()-1) ? -1 : (index+1);
  ghost_size = sizeof(DataType) * GHOST_SIZE;
  left_ghost = new Ghost(RIGHT);
  right_ghost = new Ghost(LEFT);
  neighbor_count = 2;
  if (index == 0) neighbor_count--;
  if (index == charm::chare::n-1) neighbor_count--;
  recv_count = 0;

  dim3 block_dim(BLOCK_DIM);
  dim3 grid_dim((data_size + (block_dim.x-1)) / block_dim.x);
  init_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width);
  boundary_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width);
  cudaDeviceSynchronize();

  start_tp = cuda::std::chrono::system_clock::now();
  send_ghosts();
}

__device__ void Block::send_ghosts() {
  if (left_index != -1) {
    memcpy(left_ghost->data, temperature + GHOST_SIZE, ghost_size);
    block_proxy->invoke(left_index, 1, left_ghost, sizeof(Ghost));
  }
  if (right_index != -1) {
    memcpy(right_ghost->data, temperature + block_width, ghost_size);
    block_proxy->invoke(right_index, 1, right_ghost, sizeof(Ghost));
  }
}

__device__ void Block::recv_ghosts(void* arg) {
  Ghost* gh = (Ghost*)arg;
  int dir = gh->dir;
  switch (dir) {
    case LEFT:
      memcpy(temperature, gh->data, ghost_size);
      break;
    case RIGHT:
      memcpy(temperature + GHOST_SIZE + block_width, gh->data, ghost_size);
      break;
  }

  if (++recv_count == neighbor_count) {
    recv_count = 0;
    update();
  }
}

__device__ void Block::update() {
  dim3 block_dim(BLOCK_DIM);
  dim3 grid_dim((data_size + (block_dim.x-1)) / block_dim.x);
  jacobi_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width);
  cudaDeviceSynchronize();

  if (++iter == n_iters) {
    end_tp = cuda::std::chrono::system_clock::now();
    cuda::std::chrono::duration<double> diff = end_tp - start_tp;
    printf("Chare %d completed %d iterations in %.6lf seconds\n", index, iter, diff.count());

    charm::end();
  } else {
    DataType* tmp = temperature;
    temperature = new_temperature;
    new_temperature = tmp;

    send_ghosts();
  }
}

// GPU kernels
__global__ void init_kernel(DataType* temperature, DataType* new_temperature,
                            int block_width) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_width + GHOST_SIZE*2) {
    temperature[i] = 0;
    new_temperature[i] = 0;
  }
}

__global__ void boundary_kernel(DataType* temperature, DataType* new_temperature,
                                int block_width) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < GHOST_SIZE || (i >= GHOST_SIZE + block_width && i < block_width + GHOST_SIZE*2)) {
    temperature[i] = 10;
    new_temperature[i] = 10;
  }
}

__global__ void jacobi_kernel(DataType* temperature, DataType* new_temperature,
                              int block_width) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= GHOST_SIZE && i < block_width + GHOST_SIZE) {
    DataType sum = 0;
    for (int j = -GHOST_SIZE; j <= GHOST_SIZE; j++) {
      sum += temperature[i+j];
    }
    new_temperature[i] = sum / (1 + GHOST_SIZE*2);
  }
}
