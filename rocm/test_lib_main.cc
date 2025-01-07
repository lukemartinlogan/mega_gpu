#include <hip/hip_runtime.h>

#include "test_lib.h"

// Create a rocm kernel that calls GetSize() from testLib
__global__ void my_kernel(void) {
  TestLib lib;
  lib.GetSize();
}

int main() {
  // Launch the kernel
  my_kernel<<<1, 1>>>();
  HIP_ERROR_CHECK(hipDeviceSynchronize());
  return 0;
}