//
// Created by llogan on 10/18/24.
//

#ifndef MEGAGPU__SIMPLE_LIB_H_
#define MEGAGPU__SIMPLE_LIB_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "hermes_shm/hermes_shm.h"

class SharedTest {
 public:
  static void run();
};

class MemoryManager2 {
 public:
  __host__ __device__ MemoryManager2() = default;
  template <int nothing = 0>
  __host__ __device__ int function();
};

#endif  // MEGAGPU__SIMPLE_LIB_H_
