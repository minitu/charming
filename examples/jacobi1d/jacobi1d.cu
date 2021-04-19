#include <stdio.h>
#include "jacobi1d.h"

#define BLOCK_DIM 256

__device__ charm::chare_proxy<Block>* block_proxy;

__device__ void charm::register_chares() {
  block_proxy = new charm::chare_proxy<Block>(1);
  block_proxy->add_entry_method(&Block::init);
  block_proxy->add_entry_method(&Block::send_ghosts);
  block_proxy->add_entry_method(&Block::recv_ghosts);
}

// Main
__device__ void charm::main(int argc, char** argv, size_t* argvs) {
  Block block;

  block_proxy->create(block, charm::n_pes());
  for (int i = 0; i < charm::n_pes(); i++) {
    block_proxy->invoke(i, 0);
  }
}

__global__ void init_kernel(DataType* temperature, DataType* new_temperature,
                            int block_width);
__global__ void boundary_kernel(DataType* temperature, DataType* new_temperature,
                                int block_width);

// Entry methods
__device__ void Block::init(void* arg) {
  index = charm::my_pe();
  iter = 0;
  block_width = 16;
  data_size = block_width + GHOST_SIZE*2;
  temperature = new DataType[data_size];
  new_temperature = new DataType[data_size];
  left_index = (index == 0) ? (charm::n_pes()-1) : (index-1);
  right_index = (index == charm::n_pes()-1) ? 0 : (index+1);
  ghost_size = sizeof(DataType) * GHOST_SIZE;
  left_ghost = new Ghost(RIGHT);
  right_ghost = new Ghost(LEFT);
  recv_count = 0;
  printf("Ghost bytes size: %d\n", sizeof(Ghost));

  dim3 block_dim(BLOCK_DIM);
  dim3 grid_dim((data_size + (block_dim.x-1)) / block_dim.x);
  init_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width);
  boundary_kernel<<<grid_dim, block_dim>>>(temperature, new_temperature, block_width);
  cudaDeviceSynchronize();

  printf("PE %d initialized\n", charm::my_pe());
  for (int i = 0; i < data_size; i++) {
    printf("%.3lf ", temperature[i]);
  }
  printf("\n");
  for (int i = 0; i < data_size; i++) {
    printf("%.3lf ", new_temperature[i]);
  }
  printf("\n");

  send_ghosts(nullptr);
}

__device__ void Block::send_ghosts(void* arg) {
  memcpy(left_ghost->data, temperature, ghost_size);
  memcpy(right_ghost->data, temperature + GHOST_SIZE + block_width, ghost_size);
  block_proxy->invoke(left_index, 2, left_ghost, sizeof(Ghost));
  block_proxy->invoke(right_index, 2, right_ghost, sizeof(Ghost));
}

__device__ void Block::recv_ghosts(void* arg) {
  Ghost* gh = (Ghost*)arg;
  int dir = gh->dir;
  printf("PE %d received from %s, %.3lf %.3lf\n", charm::my_pe(), (dir == 0) ? "left" : "right", gh->data[0], gh->data[1]);

  if (++recv_count == 2) {
    charm::end(charm::my_pe());
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
    temperature[i] = 1;
    new_temperature[i] = 1;
  }
}

__global__ void jacobi_kernel(DataType* temperature, DataType* new_temperature,
                              int block_width) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= GHOST_SIZE && i < block_width + GHOST_SIZE) {
    // TODO
  }
}
