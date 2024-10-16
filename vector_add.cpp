#include <cuda_runtime.h>
#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <assert.h>

// Define a simple struct to demonstrate UM usage
struct MyStruct {
  int x;
  float y;
};

// CUDA kernel that accesses shared memory
__global__ void my_kernel(MyStruct* ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 1) { // Only process the first block
    ptr->x = 100;
    ptr->y = 105;
    printf("Kernel: x=%d, y=%f\n", ptr->x, ptr->y);
  }
}

void cuda_test() {
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

  std::cout << "Result 1: x=" << hostPtr->x << ", y=" << hostPtr->y << std::endl;
  my_kernel<<<grid, block>>>(hostPtr);
  cudaDeviceSynchronize();
  std::cout << "Result 2: x=" << hostPtr->x << ", y=" << hostPtr->y << std::endl;

  // Free memory
  cudaFreeHost(hostPtr);
}

void shm_test() {
  // Create a POSIX SHM segment
  int fd = shm_open("my_shm", O_RDWR | O_CREAT, 0644);
  ftruncate(fd, 1024 * 1024); // Allocate 1MB of memory

  // Map the SHM to the host's memory
  MyStruct* shmem = (MyStruct*)mmap(NULL, 1024 * 1024,
                                    PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);

  // Pin the memory to the host's memory
  cudaHostRegister(shmem, 1024 * 1024, cudaHostRegisterPortable);

  // Launch a GPU kernel that accesses the SHM pointer
  int blockSize = 256;
  int gridSize = (1024 * 1024 + blockSize - 1) / blockSize;

  dim3 block(blockSize);
  dim3 grid(gridSize);

  shmem->x = 0;
  shmem->y = 0;
  std::cout << "Result: x=" << shmem->x << ", y=" << shmem->y << std::endl;
  my_kernel<<<grid, block>>>(shmem);
  cudaDeviceSynchronize();
  std::cout << "Result: x=" << shmem->x << ", y=" << shmem->y << std::endl;
  assert(shmem->x = 100);
  assert(shmem->y = 105);

  // Clean up
  munmap(shmem, 1024 * 1024);
  close(fd);
}

int main() {
  shm_test();
  return 0;
}

