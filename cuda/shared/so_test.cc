//
// Created by llogan on 10/18/24.
//

#include "so_test.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void TestGpuKern() { printf("yea?\n"); }

void SharedTest::run() {
  TestGpuKern<<<1, 1>>>();
  cudaDeviceSynchronize();
}