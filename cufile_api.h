/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// #endif  // CUFILE_API_H
#ifndef CUFILE_API_H
#define CUFILE_API_H

#include <cufile.h>
#include <dlfcn.h>
#include <iostream>
#include <cstdlib>

namespace hermes::adapter {

class CuFileApi {
 public:
  // Function pointers for cuFile APIs
  CUfileError_t (*cuFileHandleRegister)(CUfileHandle_t *, CUfileDescr_t *) = nullptr;
  CUfileError_t (*cuFileHandleDeregister)(CUfileHandle_t) = nullptr;
  ssize_t (*cuFileRead)(CUfileHandle_t, void *, size_t, off_t, size_t) = nullptr;
  ssize_t (*cuFileWrite)(CUfileHandle_t, const void *, size_t, off_t, size_t) = nullptr;
  CUfileError_t (*cuFileBufRegister)(void *, size_t, uint32_t) = nullptr;
  CUfileError_t (*cuFileBufDeregister)(void *) = nullptr;

  // Constructor to load symbols dynamically
  CuFileApi() {
    void *handle = dlopen("libcufile.so", RTLD_LAZY);
    if (!handle) {
      std::cerr << "Failed to load libcufile.so: " << dlerror() << std::endl;
      exit(EXIT_FAILURE);
    }

    // Load cuFile API symbols
    cuFileHandleRegister = (CUfileError_t (*)(CUfileHandle_t *, CUfileDescr_t *))dlsym(handle, "cuFileHandleRegister");
    cuFileHandleDeregister = (CUfileError_t (*)(CUfileHandle_t))dlsym(handle, "cuFileHandleDeregister");
    cuFileRead = (ssize_t (*)(CUfileHandle_t, void *, size_t, off_t, size_t))dlsym(handle, "cuFileRead");
    cuFileWrite = (ssize_t (*)(CUfileHandle_t, const void *, size_t, off_t, size_t))dlsym(handle, "cuFileWrite");
    cuFileBufRegister = (CUfileError_t (*)(void *, size_t, uint32_t))dlsym(handle, "cuFileBufRegister");
    cuFileBufDeregister = (CUfileError_t (*)(void *))dlsym(handle, "cuFileBufDeregister");

    // Check for errors
    if (!cuFileHandleRegister || !cuFileHandleDeregister || !cuFileRead || 
        !cuFileWrite || !cuFileBufRegister || !cuFileBufDeregister) {
      std::cerr << "Failed to load cuFile symbols: " << dlerror() << std::endl;
      exit(EXIT_FAILURE);
    }
  }
};

#include "hermes_shm/util/singleton.h"
// Singleton macro
#define HERMES_CUFILE_API \
  hshm::EasySingleton<::hermes::adapter::CuFileApi>::GetInstance()

}  // namespace hermes::adapter

#endif  // CUFILE_API_H
