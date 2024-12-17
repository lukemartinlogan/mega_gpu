#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  // Step 1: Initialize CUDA runtime
  cudaDeviceSynchronize();
  cudaSetDevice(0);

  std::cout << "CUDA Initialized." << std::endl;

  const int N = 1024; // Vector size
  const size_t size = N * sizeof(float);

  float *h_A, *h_B, *h_C; // Host memory
  float *d_A, *d_B, *d_C; // Device memory

  h_A = (float*)malloc(size);
  h_B = (float*)malloc(size);
  h_C = (float*)malloc(size);

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  for (int i = 0; i < N; ++i) {
    h_A[i] = i * 1.0f;
    h_B[i] = i * 2.0f;
  }

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  std::cout << "first 10 elements:" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
  }

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(h_A); free(h_B); free(h_C);

  return 0;
}

