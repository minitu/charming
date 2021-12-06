#include <stdio.h>
#include "heap.h"

__device__ __forceinline__ void swap(uint64_t* x, uint64_t* y) {
  uint64_t tmp = *x;
  *x = *y;
  *y = tmp;
}

__device__ size_t partition(uint64_t* arr, size_t lo, size_t hi) {
  uint64_t x = arr[hi];
  size_t i = lo-1;
  for (size_t j = lo; j <= hi-1; j++) {
    if (arr[j] <= x) {
      i++;
      swap(&arr[i], &arr[j]);
    }
  }
  swap(&arr[i+1], &arr[hi]);
  return (i+1);
}

__device__ void quicksort(uint64_t* arr, size_t lo, size_t hi) {
  if (lo < hi) {
    size_t p = partition(arr, lo, hi);
    quicksort(arr, lo, p-1);
    quicksort(arr, p+1, hi);
  }
}

__device__ void sort(uint64_t* arr, size_t count) {
  quicksort(arr, 0, count-1);
}

__device__ int min_heap::push(uint64_t key) {
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

__device__ uint64_t min_heap::pop() {
  if (size == 0) return UINT64_MAX;

  // Should return value at root
  uint64_t ret = buf[0];
  // Move last element to root and heapify
  buf[0] = buf[size-1];
  size--;
  size_t cur = 0;
  while (true) {
    bool left_exist = (left(cur) < size);
    bool right_exist = (right(cur) < size);
    if (!left_exist && !right_exist) break;
    uint64_t left_val = left_exist ? buf[left(cur)] : UINT64_MAX;
    uint64_t right_val = right_exist ? buf[right(cur)] : UINT64_MAX;
    bool left_smaller = left_val < right_val;
    size_t smaller_idx = left_smaller ? left(cur) : right(cur);;
    uint64_t min_val = left_smaller ? left_val : right_val;
    uint64_t cur_val = buf[cur];
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
    printf("%p ", (void*)buf[i]);
  }
  printf("\n");
}
