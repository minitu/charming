#include <stdio.h>
#include "hello.h"

__device__ void charm::register_chares() {
  // Register Hello chare and its entry methods
  charm::chare_proxies[0] = new charm::chare_proxy<Hello>(0);
  charm::entry_method**& entry_methods = static_cast<charm::chare_proxy<Hello>*>(charm::chare_proxies[0])->entry_methods;
  entry_methods = new charm::entry_method*[2];
  entry_methods[0] = new charm::entry_method_impl<Hello>(0, &Hello::greet);
}

__device__ void Hello::greet(void* arg) {
  int* recv_ints = (int*)arg;
  printf("Hello I'm %d of %d! Received %d\n", i, n, recv_ints[0]);

  if (i == n-1) {
    charm::exit();
  } else {
    int send_int[1] = {recv_ints[0] + 1};
    charm::chare_proxy<Hello>* my_chare = static_cast<charm::chare_proxy<Hello>*>(charm::chare_proxies[0]);
    my_chare->invoke(i + 1, 0, send_int, sizeof(int));
  }
}

__device__ size_t Hello::pack_size() { return 0; }
__device__ void Hello::pack(void* ptr) {}
__device__ void Hello::unpack(void* ptr) {}

// Main
__device__ void charm::main() {
  // Create and populate object that will become the basis of chares
  Hello my_obj;

  // Get a handle to the registered chare
  charm::chare_proxy<Hello>* my_chare = static_cast<charm::chare_proxy<Hello>*>(charm::chare_proxies[0]);

  // Create chares using the data in my object
  my_chare->create(my_obj, 20);

  // Invoke entry method
  int send_int[1] = {0};
  my_chare->invoke(0, 0, send_int, sizeof(int));
}
