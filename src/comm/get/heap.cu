#include <stdio.h>
#include "heap.h"

__device__ __forceinline__ void swap(composite_t* x, composite_t* y) {
  composite_t tmp = *x;
  *x = *y;
  *y = tmp;
}

__device__ int min_heap::push(const composite_t& key) {
  if (size >= max_size) return -1;

  // Insert key at the end and heapify bottom-up
  buf[size] = key;
  size_t cur = size;
  while (cur > 0) {
    if (buf[cur] < buf[parent(cur)]) {
      swap(&buf[cur], &buf[parent(cur)]);
    } else {
      break;
    }
    cur = parent(cur);
  }
  size++;

  return 0;
}

__device__ composite_t min_heap::top() {
  if (size == 0) return UINT64_MAX;

  return buf[0];
}

__device__ composite_t min_heap::pop() {
  if (size == 0) return UINT64_MAX;

  // Should return value at root
  composite_t ret = buf[0];
  // Move last element to root and heapify
  buf[0] = buf[size-1];
  size--;
  size_t cur = 0;
  while (true) {
    bool left_exist = (left(cur) < size);
    bool right_exist = (right(cur) < size);
    if (!left_exist && !right_exist) break;
    composite_t left_val = left_exist ? buf[left(cur)] : composite_t(UINT64_MAX);
    composite_t right_val = right_exist ? buf[right(cur)] : composite_t(UINT64_MAX);
    bool left_smaller = left_val < right_val;
    size_t smaller_idx = left_smaller ? left(cur) : right(cur);;
    composite_t min_val = left_smaller ? left_val : right_val;
    composite_t cur_val = buf[cur];
    if (cur_val < min_val) break;
    else {
      swap(&buf[cur], &buf[smaller_idx]);
      cur = smaller_idx;
    }
  }

  return ret;
}

__device__ void min_heap::print() {
  printf("Min-heap size: %llu, max: %llu\n", size, max_size);
  for (size_t i = 0; i < size; i++) {
    printf("%p ", (void*)buf[i].data);
  }
  printf("\n");
}
