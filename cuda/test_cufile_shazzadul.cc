// #include <cuda_runtime.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <iostream>
// #include <cstring>
// #include "cufile_api.h" 

// #define CHECK_CUDA(call)                                                       \
//     do {                                                                       \
//         cudaError_t err = call;                                                \
//         if (err != cudaSuccess) {                                              \
//             std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;\
//             exit(1);                                                           \
//         }                                                                      \
//     } while (0)

// // CUDA Kernel for vector addition
// __global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) {
//         C[i] = A[i] + B[i];
//     }
// }

// int main() {
//     const int N = 1024; // Vector size
//     const size_t size = N * sizeof(float);

//     // GPU memory
//     float *d_A, *d_B, *d_C;

//     CHECK_CUDA(cudaMalloc(&d_A, size));
//     CHECK_CUDA(cudaMalloc(&d_B, size));
//     CHECK_CUDA(cudaMalloc(&d_C, size));

//     // Register GPU memory with cuFile
//     HERMES_CUFILE_API->cuFileBufRegister(d_C, size, 0);

//     // Initialize input vectors on host

//     // float *h_A = new float[N];
//     // float *h_B = new float[N];

//     // for (int i = 0; i < N; ++i) {
//     //    d_A[i] = i * 1.0f;
//     //    d_B[i] = i * 2.0f;
//     // }
//     // // Copy input vectors to device
//     // CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
//     // CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

//     // Perform vector addition
//     vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);



//     CHECK_CUDA(cudaDeviceSynchronize());

//     // std::cout << "Vector addition completed on GPU." << std::endl;
//     // for (int i = 0; i < 5; ++i) {
//     //     std::cout << "C[" << i << "] = " << d_C[i] << std::endl;
//     // }

//     // Open output file with O_DIRECT
//     int fd_out = open("gpu_output_file.dat", O_CREAT | O_WRONLY | O_DIRECT, 0664);
//     if (fd_out < 0) {
//         std::cerr << "Failed to open output file: " << strerror(errno) << std::endl;
//         // delete[] h_A;
//         // delete[] h_B;
//         // CHECK_CUDA(cudaFree(d_A));
//         // CHECK_CUDA(cudaFree(d_B));
//         // CHECK_CUDA(cudaFree(d_C));
//         return -1;
//     }

//     CUfileHandle_t file_handle_out;
//     CUfileDescr_t cf_descr = {};
  
//     memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
//     cf_descr.handle.fd = fd_out;
//     cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
//     HERMES_CUFILE_API->cuFileHandleRegister(&file_handle_out, &cf_descr);

//     // Write result to file
//     ssize_t written_bytes = HERMES_CUFILE_API->cuFileWrite(file_handle_out, d_C, size, 0, 0);
//     if (written_bytes < 0) {
//         std::cerr << "cuFileWrite failed! Error code: " << written_bytes << std::endl;
//         return -1;
//     }
//     std::cout << "Data written to file successfully: " << written_bytes << " bytes." << std::endl;

//     close(fd_out);

//     // Cleanup
//     HERMES_CUFILE_API->cuFileHandleDeregister(file_handle_out);
//     HERMES_CUFILE_API->cuFileBufDeregister(d_C);
//     CHECK_CUDA(cudaFree(d_A));
//     CHECK_CUDA(cudaFree(d_B));
//     CHECK_CUDA(cudaFree(d_C));

//     std::cout << "Program completed successfully!" << std::endl;
//     return 0;
// }




// #include <cuda_runtime.h>
// #include <fcntl.h>
// #include <unistd.h>
// #include <iostream>
// #include <fstream> 
// #include <cstring>
// #include "cufile_api.h" 

// #define CHECK_CUDA(call)                                                       \
//     do {                                                                       \
//         cudaError_t err = call;                                                \
//         if (err != cudaSuccess) {                                              \
//             std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;\
//             exit(1);                                                           \
//         }                                                                      \
//     } while (0)

// // CUDA Kernel for vector addition
// __global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) {
//         C[i] = A[i] + B[i];
//     }
// }

// int main() {
//     const int N = 1024; // Vector size
//     const size_t size = N * sizeof(float);

//     // GPU memory
//     float *d_A, *d_B, *d_C;

//     CHECK_CUDA(cudaMalloc(&d_A, size));
//     CHECK_CUDA(cudaMalloc(&d_B, size));
//     CHECK_CUDA(cudaMalloc(&d_C, size));

//     // Register GPU memory with cuFile
//     HERMES_CUFILE_API->cuFileBufRegister(d_C, size, 0);

//     // Initialize input vectors on host
//     float *h_A = new float[N];
//     float *h_B = new float[N];
//     float *h_C = new float[N]; // Host buffer for validation

//     for (int i = 0; i < N; ++i) {
//         h_A[i] = i * 1.0f;
//         h_B[i] = i * 2.0f;
//     }

//     // Copy input vectors to device
//     CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
//     CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

//     // Perform vector addition
//     vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

//     CHECK_CUDA(cudaDeviceSynchronize());

//     std::cout << "Vector addition completed on GPU." << std::endl;

//     // Open output file with O_DIRECT
//     int fd_out = open("gpu_output_file.dat", O_CREAT | O_WRONLY | O_DIRECT, 0664);
//     if (fd_out < 0) {
//         std::cerr << "Failed to open output file: " << strerror(errno) << std::endl;
//         delete[] h_A;
//         delete[] h_B;
//         delete[] h_C;
//         CHECK_CUDA(cudaFree(d_A));
//         CHECK_CUDA(cudaFree(d_B));
//         CHECK_CUDA(cudaFree(d_C));
//         return -1;
//     }

//     CUfileHandle_t file_handle_out;
//     CUfileDescr_t cf_descr = {};

//     memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
//     cf_descr.handle.fd = fd_out;
//     cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
//     HERMES_CUFILE_API->cuFileHandleRegister(&file_handle_out, &cf_descr);

//     // Write result to file
//     ssize_t written_bytes = HERMES_CUFILE_API->cuFileWrite(file_handle_out, d_C, size, 0, 0);
//     if (written_bytes < 0) {
//         std::cerr << "cuFileWrite failed! Error code: " << written_bytes << std::endl;
//         return -1;
//     }
//     std::cout << "Data written to file successfully: " << written_bytes << " bytes." << std::endl;

//     close(fd_out);
//     HERMES_CUFILE_API->cuFileHandleDeregister(file_handle_out);

//     // Read back the data for verification
//     int fd_in = open("gpu_output_file.dat", O_RDONLY | O_DIRECT);
//     if (fd_in < 0) {
//         std::cerr << "Failed to open input file: " << strerror(errno) << std::endl;
//         return -1;
//     }

//     CUfileHandle_t file_handle_in;
//     cf_descr.handle.fd = fd_in;
//     HERMES_CUFILE_API->cuFileHandleRegister(&file_handle_in, &cf_descr);

//     ssize_t read_bytes = HERMES_CUFILE_API->cuFileRead(file_handle_in, d_C, size, 0, 0);
//     if (read_bytes < 0) {
//         std::cerr << "cuFileRead failed! Error code: " << read_bytes << std::endl;
//         return -1;
//     }

//     CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

//     // Validate data
//     for (int i = 0; i < 5; ++i) {
//         std::cout << "Data read from output file ." << std::endl;
//         std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
//     }

//     std::cout << "Data read back successfully and validated." << std::endl;

//     close(fd_in);
//     HERMES_CUFILE_API->cuFileHandleDeregister(file_handle_in);

//     // Cleanup
//     delete[] h_A;
//     delete[] h_B;
//     delete[] h_C;
//     HERMES_CUFILE_API->cuFileBufDeregister(d_C);
//     CHECK_CUDA(cudaFree(d_A));
//     CHECK_CUDA(cudaFree(d_B));
//     CHECK_CUDA(cudaFree(d_C));

//     std::cout << "Program completed successfully!" << std::endl;
//     return 0;
// }

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <fstream> 
#include <cstring>
#include "cufile_api.h" 

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;\
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// CUDA Kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1024; // Vector size
    const size_t size = N * sizeof(float);

    // GPU memory
    float *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    // Register GPU memory with cuFile
    HERMES_CUFILE_API->cuFileBufRegister(d_C, size, 0);

    // Initialize input vectors on host
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N]; // Host buffer for validation

    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Copy input vectors to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Perform vector addition
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy GPU data back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    std::cout << "Result Vector Addition:" << std::endl;

    // Print data after vector addition before writing to file
    for (int i = 0; i < 5; ++i) {
        std::cout << " C[" << i << "] = " << h_C[i] << std::endl;
    }

    std::cout << "Vector addition completed on GPU." << std::endl;

    // Open output file with O_DIRECT
    int fd_out = open("gpu_output_file.dat", O_CREAT | O_WRONLY | O_DIRECT, 0664);
    if (fd_out < 0) {
        std::cerr << "Failed to open output file:" << strerror(errno) << std::endl;
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        return -1;
    }

    CUfileHandle_t file_handle_out;
    CUfileDescr_t cf_descr = {};

    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = fd_out;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    HERMES_CUFILE_API->cuFileHandleRegister(&file_handle_out, &cf_descr);

    // Write result to file
    ssize_t written_bytes = HERMES_CUFILE_API->cuFileWrite(file_handle_out, d_C, size, 0, 0);
    if (written_bytes < 0) {
        std::cerr << "cuFileWrite failed! Error code: " << written_bytes << std::endl;
        return -1;
    }
    std::cout << "Data written to file successfully: " << written_bytes << " bytes." << std::endl;

    close(fd_out);
    HERMES_CUFILE_API->cuFileHandleDeregister(file_handle_out);

    // Allocate a new GPU memory buffer for reading data back
    float *d_C_read;
    CHECK_CUDA(cudaMalloc(&d_C_read, size));

    // Read back the data for verification
    int fd_in = open("gpu_output_file.dat", O_RDONLY | O_DIRECT);
    if (fd_in < 0) {
        std::cerr << "Failed to open input file: " << strerror(errno) << std::endl;
        CHECK_CUDA(cudaFree(d_C_read));
        return -1;
    }

    CUfileHandle_t file_handle_in;
    cf_descr.handle.fd = fd_in;
    HERMES_CUFILE_API->cuFileHandleRegister(&file_handle_in, &cf_descr);

    ssize_t read_bytes = HERMES_CUFILE_API->cuFileRead(file_handle_in, d_C_read, size, 0, 0);
    if (read_bytes < 0) {
        std::cerr << "cuFileRead failed! Error code: " << read_bytes << std::endl;
        CHECK_CUDA(cudaFree(d_C_read));
        return -1;
    }

    float *h_C_read = new float[N]; // Host buffer for read-back data
    CHECK_CUDA(cudaMemcpy(h_C_read, d_C_read, size, cudaMemcpyDeviceToHost));

    std::cout << "Reading data from output file:" << std::endl;
    // Validate data
    for (int i = 0; i < 5; ++i) {
        std::cout << "C_read[" << i << "] = " << h_C_read[i] << std::endl;
    }

    std::cout << "Data read back successfully and validated." << std::endl;

    close(fd_in);
    HERMES_CUFILE_API->cuFileHandleDeregister(file_handle_in);

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_read;
    HERMES_CUFILE_API->cuFileBufDeregister(d_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_read));

    std::cout << "Program completed successfully!" << std::endl;
    return 0;
}
