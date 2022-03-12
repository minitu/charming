#include <iostream>
#include <unistd.h>
#include <chrono>
#include "jacobi_kernel.h"

#define cudaCheckError() {                                                           \
  cudaError_t e = cudaGetLastError();                                                \
  if (cudaSuccess != e) {                                                            \
    printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(-1);                                                                        \
  }                                                                                  \
}

#define DIVIDEBY3 0.333333
#define DIVIDEBY5 0.2
#define DIVIDEBY7 0.142857

__global__ void jacobi_1d_kernel(DataType* temp, DataType* new_temp,
    int width) {
  int i = (blockDim.x*blockIdx.x+threadIdx.x)+1;
  if (i <= width) {
    new_temp[i] = (temp[i] + temp[i-1] + temp[i+1]) * DIVIDEBY3;
  }
}


__global__ void jacobi_2d_kernel(DataType* temp, DataType* new_temp,
    int width, int height) {
  int i = (blockDim.x*blockIdx.x+threadIdx.x)+1;
  int j = (blockDim.y*blockIdx.y+threadIdx.y)+1;
  if (i <= width && j <= height) {
    new_temp[IDX_2D(i,j)] = (temp[IDX_2D(i,j)] + temp[IDX_2D(i-1,j)]
        + temp[IDX_2D(i+1,j)] + temp[IDX_2D(i,j-1)] + temp[IDX_2D(i,j+1)]) * DIVIDEBY5;
  }
}

__global__ void jacobi_3d_kernel(DataType* temp, DataType* new_temp,
    int width, int height, int depth) {
  int i = (blockDim.x*blockIdx.x+threadIdx.x)+1;
  int j = (blockDim.y*blockIdx.y+threadIdx.y)+1;
  int k = (blockDim.z*blockIdx.z+threadIdx.z)+1;

  if (i <= width && j <= height && k <= depth) {
    new_temp[IDX_3D(i,j,k)] = (temp[IDX_3D(i,j,k)] +
      temp[IDX_3D(i-1,j,k)] + temp[IDX_3D(i+1,j,k)] +
      temp[IDX_3D(i,j-1,k)] + temp[IDX_3D(i,j+1,k)] +
      temp[IDX_3D(i,j,k-1)] + temp[IDX_3D(i,j,k+1)]) * DIVIDEBY7;
  }
}

Block::Block(int width_, int height_, int depth_, int n_iters_, int warmup_iters_) {
  // Store parameters
  width = width_;
  height = height_;
  depth = depth_;
  n_iters = n_iters_;
  warmup_iters = warmup_iters_;

  // Determine number of dimensions
  dims = 1;
  if (height > 0) dims++;
  if (depth > 0) dims++;

  // Print configuration
  std::cout << "Config: Block size " << width << " x " << height << " x "
    << depth << " (" << dims << "D), Iters: " << n_iters << ", Warmup: "
    << warmup_iters << std::endl;


  // Allocate memory
  temp = nullptr;
  new_temp = nullptr;
  size_t block_size = (width+2) * (height+2) * (depth+2) * sizeof(DataType);
  cudaMalloc(&temp, block_size);
  cudaMalloc(&new_temp, block_size);
  cudaCheckError();

  // Create CUDA stream
  cudaStreamCreate(&stream);
  cudaCheckError();

  // Memset
  cudaMemsetAsync(temp, 1, block_size, stream);
  cudaMemsetAsync(new_temp, 1, block_size, stream);
}

Block::~Block() {
  // Free memory
  cudaFree(temp);
  cudaFree(new_temp);
  cudaCheckError();

  // Destroy CUDA stream
  cudaStreamDestroy(stream);
}

void Block::run() {
  std::chrono::time_point<std::chrono::system_clock> start, end;

  for (int i = 0; i < n_iters + warmup_iters; i++) {
    if (i == warmup_iters) {
      start = std::chrono::system_clock::now();
    }

    // Invoke Jacobi kernel
    if (dims == 1) {
      dim3 block_dim(1024);
      dim3 grid_dim((width+block_dim.x-1)/block_dim.x);

      jacobi_1d_kernel<<<grid_dim, block_dim, 0, stream>>>(
          temp, new_temp, width);
    } else if (dims == 2) {
      dim3 block_dim(32,32);
      dim3 grid_dim((width+block_dim.x-1)/block_dim.x,
          (height+block_dim.y-1)/block_dim.y);

      jacobi_2d_kernel<<<grid_dim, block_dim, 0, stream>>>(
          temp, new_temp, width, height);
    } else if (dims == 3) {
      dim3 block_dim(8,8,8);
      dim3 grid_dim((width+block_dim.x-1)/block_dim.x,
          (height+block_dim.y-1)/block_dim.y,
          (depth+block_dim.z-1)/block_dim.z);

      jacobi_3d_kernel<<<grid_dim, block_dim, 0, stream>>>(
          temp, new_temp, width, height, depth);
    }
  }
  cudaStreamSynchronize(stream);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
  std::cout << "Average iteration time: " << elapsed.count() / n_iters * 1e6
    << " us" << std::endl;
}

int main(int argc, char** argv) {
  int width = 1024;
  int height = -1;
  int depth = -1;
  int n_iters = 100;
  int warmup_iters = 10;

  // Process command line arguments
  int c;
  while ((c = getopt(argc, argv, "x:y:z:i:w:")) != -1) {
    switch(c) {
      case 'x':
        width = atoi(optarg);
        break;
      case 'y':
        height = atoi(optarg);
        break;
      case 'z':
        depth = atoi(optarg);
        break;
      case 'i':
        n_iters = atoi(optarg);
        break;
      case 'w':
        warmup_iters = atoi(optarg);
        break;
      default:
        std::cerr << "Invalid argument" << std::endl;
        exit(-1);
    }
  }

  if (!((width > 0 && height == -1 && depth == -1)
        || (width > 0 && height > 0 && depth == -1)
        || (width > 0 && height > 0 && depth > 0))) {
    std::cerr << "Invalid dimensions: " << width << " x " << height << " x "
      << depth << std::endl;
  }

  // Create block object
  std::cout << "Creating block object..." << std::endl;
  Block block(width, height, depth, n_iters, warmup_iters);

  // Execute Jacobi iterations
  block.run();

  return 0;
}
