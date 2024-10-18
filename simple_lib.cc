//
// Created by llogan on 10/18/24.
//

#include "simple_lib.h"

__host__ __device__
int MyClass::foo() {
#ifndef __CUDA_ARCH__
  return 42;
#else
  return 100;
#endif
}