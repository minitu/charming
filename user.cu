#include <stdio.h>
#include "user.h"

__device__ void hello() {
  printf("Hello!\n");
}

__device__ int fibonacci(int n) {
  if (n == 1) return 0;
  else if (n == 2) return 1;
  else return fibonacci(n-2) + fibonacci(n-1);
}

__device__ void register_entry_methods(int* entry_methods) {
  // TODO
}
