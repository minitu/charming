#include <stdio.h>
#include "user.h"

__device__ void register_chare_types(ChareType** chare_types) {
  // Register Foo and its entry methods
  chare_types[0] = new Chare<Foo>(0);
  EntryMethod**& entry_methods = static_cast<Chare<Foo>*>(chare_types[0])->entry_methods;
  entry_methods = new EntryMethod*[2];
  entry_methods[0] = new EntryMethodImpl<Foo, void(Foo&)>(0, &Foo::hello);
  entry_methods[1] = new EntryMethodImpl<Foo, void(Foo&)>(1, &Foo::morning);

  // Register Bar and its entry methods
  chare_types[1] = new Chare<Bar>(1);
  entry_methods = static_cast<Chare<Bar>*>(chare_types[1])->entry_methods;
  entry_methods = new EntryMethod*[2];
  entry_methods[0] = new EntryMethodImpl<Bar, void(Bar&)>(0, &Bar::hammer);
}

// Foo
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
  printf("Foo packed %d at ptr %p\n", *(int*)ptr, ptr);
}

__device__ void Foo::unpack(void* ptr) {
  i = *(int*)ptr;
  printf("Foo unpacked %d from ptr %p\n", i, ptr);
}

// Bar
__device__ void Bar::hammer() {
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
__device__ void charm_main(ChareType** chare_types) {
  Foo my_obj(1);
  Chare<Foo>* my_chare = static_cast<Chare<Foo>*>(chare_types[0]);
  my_chare->create(my_obj);
}
