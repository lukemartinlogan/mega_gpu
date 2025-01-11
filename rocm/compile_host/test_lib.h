#ifdef WITH_HIP
#include <hip/hip_runtime.h>
#define HOST_DEV __host__ __device__
#else
#define HOST_DEV
#endif

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
  HOST_DEV size_t GetSize();
};