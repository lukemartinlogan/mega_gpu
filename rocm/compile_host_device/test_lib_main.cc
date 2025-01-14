#include <hip/hip_runtime.h>

#include "test_lib.h"
#include "test_lib2.h"

// Create a rocm kernel that calls GetSize() from testLib
__global__ void my_kernel(void) {
  TestLib lib;
  lib.GetSize();
}

int main() {
  // Launch the kernel
  my_kernel<<<1, 1>>>();
  HIP_ERROR_CHECK(hipDeviceSynchronize());

  TestLib2 lib2;
  lib2.Try();
  return 0;
}