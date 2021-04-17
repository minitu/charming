#include <stdio.h>
#include "simple.h"

__device__ charm::chare_proxy<Foo>* foo_proxy;
__device__ charm::chare_proxy<Bar>* bar_proxy;

__device__ void charm::register_chares() {
  // Register chares and entry methods
  charm::chare_proxies[0] = foo_proxy = new charm::chare_proxy<Foo>(0, 2);
  foo_proxy->add_entry_method(&Foo::hello);
  foo_proxy->add_entry_method(&Foo::morning);

  // Register Bar and its entry methods
  charm::chare_proxies[1] = bar_proxy = new charm::chare_proxy<Bar>(1, 1);
  bar_proxy->add_entry_method(&Bar::hammer);
}

// Foo
__device__ void Foo::hello(void* arg) {
  printf("Hello! My int is %d\n", my_int);
}

__device__ void Foo::morning(void* arg) {
  int* recv_ints = (int*)arg;
  printf("Good morning! Received %d and %d\n", *recv_ints, *(recv_ints+1));
}

__device__ size_t Foo::pack_size() {
  return sizeof(int);
}

__device__ void Foo::pack(void* ptr) {
  *(int*)ptr = my_int;
}

__device__ void Foo::unpack(void* ptr) {
  my_int = *(int*)ptr;
}

// Bar
__device__ void Bar::hammer(void* arg) {
  printf("Hammer!\n");
}

__device__ size_t Bar::pack_size() {
  return sizeof(char);
}

__device__ void Bar::pack(void* ptr) {
  *(char*)ptr = ch;
}

__device__ void Bar::unpack(void* ptr) {
  ch = *(char*)ptr;
}

// Main
__device__ void charm::main() {
  // Create and populate object that will become the basis of chares
  Foo my_obj(1);

  // Create Foo chares using the data in my object
  foo_proxy->create(my_obj, 20);

  // Invoke entry methods (chare index, entry method index, source buffer, buffer size)
  foo_proxy->invoke(6 /* Chare index */, 0 /* Entry method index */);
  int a[2] = {10, 11};
  foo_proxy->invoke(11, 1, a, sizeof(int) * 2);

  // Send termination messages to all PEs
  charm::exit();
}
