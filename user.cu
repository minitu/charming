#include <stdio.h>
#include "user.h"

__device__ void hello() {
  printf("Hello!\n");
}

__device__ void morning() {
  printf("Good morning!\n");
}

__device__ void register_entry_methods(EntryMethod** entry_methods) {
  entry_methods[0] = new EntryMethodImpl<void()>(hello);
  entry_methods[1] = new EntryMethodImpl<void()>(morning);
}

__device__ void Foo::hello() {
  printf("Hello!\n");
}

__device__ void Foo::morning() {
  printf("Good morning!\n");
}

__device__ void charm_main() {
  Foo a(9);
  Chare<Foo> a_chare(a);
  a_chare.invoke(0, 0);
  a_chare.invoke(1, 1);
  a_chare.invoke(2, 0);
}
