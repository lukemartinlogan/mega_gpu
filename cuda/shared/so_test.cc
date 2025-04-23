//
// Created by llogan on 10/18/24.
//

#include "so_test.h"

#include <cuda_runtime.h>

void SharedTest::run() {
  TestGpuKern<<<1, 1>>>();
  cudaDeviceSynchronize();
}