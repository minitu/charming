#include <stdio.h>
#include "jacobi1d.h"

#define BLOCK_DIM 256

__device__ charm::chare_proxy<Block>* block_proxy;

__device__ void charm::register_chares() {
  block_proxy = new charm::chare_proxy<Block>(1);
  block_proxy->add_entry_method(&Block::init);
  block_proxy->add_entry_method(&Block::send_halo);
}

// Main
__device__ void charm::main(int argc, char** argv, size_t* argvs) {
  Block block;

  block_proxy->create(block, charm::n_pes());
  for (int i = 0; i < charm::n_pes(); i++) {
    block_proxy->invoke(i, 0);
  }

  charm::exit();
}

__global__ void init_kernel(DataType* temperature, int block_width);

// Entry methods
__device__ void Block::init(void* arg) {
  int block_width = 1024;
  int data_size = block_width + GHOST_SIZE*2;
  temperature = new DataType[data_size];

  dim3 block_dim(BLOCK_DIM);
  dim3 grid_dim((data_size + (block_dim.x-1)) / block_dim.x);
  init_kernel<<<grid_dim, block_dim>>>(temperature, block_width);
  cudaDeviceSynchronize();
}

__device__ void Block::send_halo(void* arg) {
  // TODO
}

// GPU kernels
__global__ void init_kernel(DataType* temperature, int block_width) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < block_width + GHOST_SIZE*2) {
    temperature[i] = 0;
  }
}

__global__ void left_boundary_kernel(DataType* temperature, int block_width) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < GHOST_SIZE) {
    temperature[i] = 1;
  }
}

__global__ void right_boundary_kernel(DataType* temperature, int block_width) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i > GHOST_SIZE + block_width && i < block_width + GHOST_SIZE*2) {
    temperature[i] = 1;
  }
}

__global__ void jacobi_kernel(DataType* temperature, DataType* new_temperature,
                              int block_width) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= GHOST_SIZE && i < block_width + GHOST_SIZE) {
    // TODO
  }
}
