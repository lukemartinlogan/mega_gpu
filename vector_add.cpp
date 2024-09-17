#include <cuda_runtime.h>
#include <iostream>

// Define a simple struct to demonstrate UM usage
struct MyStruct {
  int x;
  float y;
};

// CUDA kernel that accesses shared memory
__global__ void my_kernel(MyStruct* ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) { // Only process the first block
    printf("Kernel: x=%d, y=%f\n", ptr->x, ptr->y);
  }
}

int main() {
  // Initialize CUDA runtime
  cudaDeviceSynchronize();
  cudaSetDevice(0);

  // Allocate memory on the host and device using UM
  size_t size = sizeof(MyStruct);
  MyStruct* hostPtr = nullptr;
  cudaHostAlloc(&hostPtr, size, cudaHostAllocMapped);

  // Create a MyStruct instance and copy it to both host and device memory
  MyStruct myStruct;
  myStruct.x = 10;
  myStruct.y = 3.14f;
  cudaMemcpy(hostPtr, &myStruct, size, cudaMemcpyHostToHost);

  // Launch a CUDA kernel that accesses the shared memory
  int blockSize = 256;
  int numBlocks = 1;
  dim3 block(blockSize);
  dim3 grid(numBlocks);

  my_kernel<<<grid, block>>>(hostPtr);

  // Wait for kernel completion
  cudaDeviceSynchronize();

  // Copy data back from device to host using UM
  std::cout << "Result: x=" << myStruct.x << ", y=" << myStruct.y << std::endl;

  // Free memory
  cudaFreeHost(hostPtr);

  return 0;
}

