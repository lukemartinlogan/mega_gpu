#include <iostream>
#include <cuda_runtime.h>
#include <stddef.h>
using namespace std;

__global__ void addVector(int *a, int *b, int *c, int n)
{	
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if( id < n ){
		c[id]=a[id]+b[id];
	}
}
// code in host(CPU)
int main (){
	
	
	//size_t bytes = SIZE*sizeof(int);
	int SIZE = 32;
	size_t bytes = SIZE * sizeof(int);
	int *a = new int [SIZE];
	int *b = new int [SIZE];
	int *c = new int [SIZE];


	for(int i=0; i<SIZE; i++){
		a[i]=i+1;
		b[i]=i+1;
	}	
	//allocating vectores in device memory(GPU)
	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, bytes);
	cudaMalloc((void **)&d_b, bytes);
	cudaMalloc((void **)&d_c, bytes);
	
	//copying vectore from CPU to GPU
	cudaMemcpy(d_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);
	
	cout<<"Invoking kernel cuda...\n"<<endl;
	
	//invoking kernel
	int threadsPerBlock = 16;
	int blocksPerGrid = (SIZE + threadsPerBlock-1)/threadsPerBlock;
	
	addVector<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b,	d_c, SIZE);

	//copy result from device(GPU) to host(CPU). C contains result in device
	cudaMemcpy(c, d_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	
	for (int i=0; i < SIZE; i++){
		cout<<"C["<<i<<"]= "<<c[i]<<endl;
	}

	//free mem of gpu
	cudaFree(d_a);	
	cudaFree(d_b);
	cudaFree(d_c);

	//free host memp(cpu)

	delete[] a;
	delete[] b;
	delete[] c;	
	
	cudaDeviceSynchronize();
	return 0;
}
	
