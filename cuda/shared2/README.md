
# CUDA Doesn't Always Like Templates

Just in case it is of use for anyone else, I encountered a very random issue when I was using templated cuda kernels. I was developing a header-only library for building custom memory allocators for CUDA programs. Won't go into detail why -- something something simulation performance. Attached is a working source code to reproduce the problem. I had divided a MemoryManager class into two header files: one for definition and the other for implemenation. In the implementation, I had used templates to remove the chance of "multiple definitions" from class functions. For class member functions, it was a necessity. 

However, I decided to also add them to the CUDA kernels (not in the class) for consistency (not correctness). The template was not used in the kernel's logic. However, confusingly, this code resulted in "invalid device" errors. I thought it was likely relating to something basic like forgetting to turn on position-independent code or using the wrong cuda architecture, but it wasn't these things. 

The real solution was simple: just remove the template from the cuda kernels and everything worked just fine. I thought it was just a very strange thing and caused me hours of headache trying to narrow down why I was encountering "invalid device" errors. Morally, I don't think this should be the case and may cause other issues with templating kernels.

## Building + Running The Error

```cmake
mkdir build
cd build
cmake ../ -DCMAKE_CUDA_ARCHITECTURES=89
make
compute-sanitizer ./test_shared
```

## Code Walkthrough

Definition header: so_test_.h
```cpp
class MemoryManager2 {
 public:
  __host__ __device__ MemoryManager2() = default;
  template <int nothing = 0>
  __host__ __device__ int function();
};
```

Implementation header: so_test.h
```cpp
template <int nothing = 0>  // NOTE(llogan): REMOVE ME AND I'LL WORK!!!
static __global__ void TestGpuKern2() {
  MemoryManager2 mngr;
  mngr.function();
  printf("IN KERNEL!!!\n");
}

template <int>
__host__ __device__ int MemoryManager2::function() {
#ifndef __CUDA_ARCH__
  int left = 0;
  for (int i = 0; i < 1024; ++i) {
    left += i << 4;
    left ^= 31;
    left += 25;
  }
  TestGpuKern2<<<1, 1>>>();
  cudaDeviceSynchronize();
  return left;
#else
  return 0;
#endif
}
```

Source: test_shared.cc
```cpp
int main() {
  printf("HERE!!!\n");
  MemoryManager2 mngr;
  mngr.function<0>();

  return 0;
}
```
