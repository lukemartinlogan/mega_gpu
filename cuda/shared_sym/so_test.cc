//
// Created by llogan on 10/18/24.
//

#include "so_test.h"

#include <cuda_runtime.h>

static __global__ void TestGpuKern() { printf("yea?\n"); }

extern "C" {
void SharedTestRun() {
  TestGpuKern<<<1, 1>>>();
  cudaDeviceSynchronize();
}
}