#include <stdio.h>
#include "user.h"

__device__ void register_chare_types(ChareType** chare_types) {
  // Register Foo and its entry methods
  chare_types[0] = new Chare<Foo>(0);
  EntryMethod**& entry_methods = static_cast<Chare<Foo>*>(chare_types[0])->entry_methods;
  entry_methods = new EntryMethod*[2];
  entry_methods[0] = new EntryMethodImpl<void(Foo&)>(0, &Foo::hello);
  entry_methods[1] = new EntryMethodImpl<void(Foo&)>(1, &Foo::morning);

  // Register Bar and its entry methods
  chare_types[1] = new Chare<Bar>(1);
  entry_methods = static_cast<Chare<Bar>*>(chare_types[1])->entry_methods;
  entry_methods = new EntryMethod*[2];
  entry_methods[0] = new EntryMethodImpl<void(Bar&)>(0, &Bar::hammer);
}

__device__ void Foo::hello() {
  printf("Hello! My int is %d\n", i);
}

__device__ void Foo::morning() {
  printf("Good morning!\n");
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

__device__ void Bar::hammer() {
  printf("Hammer!\n");
}

__device__ void charm_main() {
  Foo my_obj;
  Chare<Foo>::create(my_obj, -1);
}
