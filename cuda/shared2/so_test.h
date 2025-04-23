#include "so_test_.h"

// NOTE(llogan): REMOVE ME AND I'LL WORK!!!
template <int nothing = 0>
static __global__ void TestGpuKern2() {
  MemoryManager2 mngr;
  mngr.function();
  printf("IN KERNEL!!!\n");
}

template <int>
__host__ __device__ int MemoryManager2::function() {
#ifndef __CUDA_ARCH__
  int left = 0;
  for (int i = 0; i < 1024; ++i) {
    left += i << 4;
    left ^= 31;
    left += 25;
  }
  TestGpuKern2<<<1, 1>>>();
  cudaDeviceSynchronize();
  return left;
#else
  return 0;
#endif
}
