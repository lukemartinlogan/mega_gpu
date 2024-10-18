//
// Created by llogan on 10/18/24.
//

#ifndef MEGAGPU__SIMPLE_LIB_H_
#define MEGAGPU__SIMPLE_LIB_H_

#include <cuda_runtime.h>

class MyClass {
 public:
  __host__ __device__ int foo();
};

#endif //MEGAGPU__SIMPLE_LIB_H_
