//
// Created by llogan on 10/18/24.
//

#ifndef MEGAGPU__SIMPLE_LIB_H_
#define MEGAGPU__SIMPLE_LIB_H_

#include <cuda_runtime.h>

class MyCudaStruct {
 public:
  int x;
  int y;
 public:
 __host__ __device__ MyCudaStruct();
  __host__ __device__ int foo();
};

#endif //MEGAGPU__SIMPLE_LIB_H_
