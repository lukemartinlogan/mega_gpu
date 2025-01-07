//
// Created by llogan on 10/18/24.
//

#include "simple_lib.h"

__host__ __device__
MyCudaStruct::MyCudaStruct() : x(0), y(0) {}

__host__ __device__
int MyCudaStruct::foo() {
#ifndef __CUDA_ARCH__
  return 42;
#else
  return 100;
#endif
}