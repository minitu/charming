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

__device__ void charm_main() {
  Foo a;
  a.create(10);
  a.invoke(1,-1);
  a.invoke(2,3);
  a.invoke(3,5);
  a.invoke(-1,-1);
}

__device__ void Foo::hello() {
  printf("Hello!\n");
}
