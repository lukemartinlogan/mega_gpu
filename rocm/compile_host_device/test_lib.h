#ifndef TEST_LIB_H
#define TEST_LIB_H

#include <hip/hip_runtime.h>

#include <cstddef>

/** Error checking for ROCM */
#define HIP_ERROR_CHECK(X)                                           \
  do {                                                               \
    if (X != hipSuccess) {                                           \
      hipError_t hipErr = hipGetLastError();                         \
      printf("HIP Error %d: %s", hipErr, hipGetErrorString(hipErr)); \
      exit(1);                                                       \
    }                                                                \
  } while (false)

class TestLib {
 public:
  __host__ __device__ size_t GetSize();
};

#endif  // TEST_LIB_H