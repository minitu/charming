#include <stdio.h>
#include "jacobi1d.h"

__device__ charm::chare_proxy<Block>* block_proxy;

__device__ void charm::register_chares() {
  block_proxy = new charm::chare_proxy<Block>(1);
  block_proxy->add_entry_method(&Block::foo);
}

__device__ void Block::foo(void* arg) {
  // TODO
  printf("Hello from PE %d\n", charm::my_pe());
}

// Main
__device__ void charm::main(int argc, char** argv, size_t* argvs) {
  Block block;
  block_proxy->create(block, charm::n_pes());

  block_proxy->invoke(charm::my_pe() + 1, 0);

  charm::exit();
}
