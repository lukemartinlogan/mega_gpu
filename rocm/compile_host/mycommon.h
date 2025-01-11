#ifndef MYCOMMON_H
#define MYCOMMON_H

#if defined(WITH_HIP)
#include <hip/hip_runtime.h>
#define HOST_DEV __host__ __device__
#else
#define HOST_DEV
#endif

/** Function content selector for CUDA */
#ifdef __CUDA_ARCH__
#define HSHM_IS_CUDA_GPU
#endif

/** Function content selector for ROCm */
#ifdef __HIP_DEVICE_COMPILE__
#define HSHM_IS_ROCM_GPU
#endif

/** Function content selector for CPU vs GPU */
#if defined(HSHM_IS_CUDA_GPU) || defined(HSHM_IS_ROCM_GPU)
#define HSHM_IS_GPU
#else
#define HSHM_IS_HOST
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

#endif  // MYCOMMON_H