#include <stdio.h>
#include "simple.h"

__device__ void charm::register_chare_types() {
  // Register Foo and its entry methods
  chare_types[0] = new charm::chare<Foo>(0);
  charm::entry_method**& foo_entry_methods = static_cast<charm::chare<Foo>*>(chare_types[0])->entry_methods;
  foo_entry_methods = new charm::entry_method*[2];
  foo_entry_methods[0] = new charm::entry_method_impl<Foo>(0, &Foo::hello);
  foo_entry_methods[1] = new charm::entry_method_impl<Foo>(1, &Foo::morning);

  // Register Bar and its entry methods
  chare_types[1] = new charm::chare<Bar>(1);
  charm::entry_method**& bar_entry_methods = static_cast<charm::chare<Bar>*>(chare_types[1])->entry_methods;
  bar_entry_methods = new charm::entry_method*[1];
  bar_entry_methods[0] = new charm::entry_method_impl<Bar>(0, &Bar::hammer);
}

// Foo
__device__ void Foo::hello(void* arg) {
  printf("Hello! My int is %d\n", i);
}

__device__ void Foo::morning(void* arg) {
  int* recv_ints = (int*)arg;
  printf("Good morning! Received %d and %d\n", *recv_ints, *(recv_ints+1));
}

__device__ size_t Foo::pack_size() {
  return sizeof(int);
}

__device__ void Foo::pack(void* ptr) {
  *(int*)ptr = i;
}

__device__ void Foo::unpack(void* ptr) {
  i = *(int*)ptr;
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

  // Get a handle to the registered Foo chare
  charm::chare<Foo>* my_chare = static_cast<charm::chare<Foo>*>(chare_types[0]);

  // Create chares using the data in my object
  my_chare->create(my_obj, 20);

  // Invoke entry methods (chare index, entry method index, source buffer, buffer size)
  my_chare->invoke(6 /* Chare index */, 0 /* Entry method index */);
  int a[2] = {10, 11};
  my_chare->invoke(11, 1, a, sizeof(int) * 2);

  // Send termination messages to all PEs
  charm::exit();
}
