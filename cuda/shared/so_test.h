#include "so_test_.h"

template <int nothing = 0>
static __global__ void TestGpuKern() {
  printf("yea?\n");
}

template <int nothing = 0>
static __global__ void TestGpuKern2() {
  MemoryManager2 mngr;
  mngr.function();
}

template <int>
HSHM_CROSS_FUN int MemoryManager2::function() {
#ifdef HSHM_IS_HOST
  int left = 0;
  for (int i = 0; i < 1024; ++i) {
    left += i << 4;
    left ^= 31;
    left += 25;
  }
  TestGpuKern2<<<1, 1>>>();
  hshm::GpuApi::Synchronize();
  return left;
#else
  return 0;
#endif
}
