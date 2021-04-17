#include <stdio.h>
#include "hello.h"

__device__ charm::chare_proxy<Hello>* hello_proxy;

__device__ void charm::register_chares() {
  // Register Hello chare and its entry methods
  hello_proxy = new charm::chare_proxy<Hello>(0, 1);
  hello_proxy->add_entry_method(&Hello::greet);
}

__device__ void Hello::greet(void* arg) {
  int recv_int = ((int*)arg)[0];
  printf("Hello I'm %d of %d! Received %d\n", i, n, recv_int);

  if (i == n-1) {
    charm::exit();
  } else {
    int send_int[1] = {recv_int + 1};
    hello_proxy->invoke(i + 1, 0, send_int, sizeof(int));
  }
}

// Main
__device__ void charm::main() {
  // Create and populate object that will become the basis of chares
  Hello my_obj;

  // Create chares using the data in my object
  hello_proxy->create(my_obj, 20);

  // Invoke entry method
  int send_int[1] = {0};
  hello_proxy->invoke(0, 0, send_int, sizeof(int));
}
