

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

#ifndef HERMES_ADAPTER_STDIO_H
#define HERMES_ADAPTER_STDIO_H

#include <string>
#include <dlfcn.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cufile.h>
#include "hermes_shm/util/singleton.h"
#include "real_api.h"
#include "hermes_shm/util/logging.h"


extern "C" {
typedef FILE * (*fopen_t)(const char * path, const char * mode);
typedef FILE * (*fopen64_t)(const char * path, const char * mode);
}
namespace hermes::adapter {

/** CuFileApi class */
class CuFileApi {
 public:
  CUfileError_t (*cuFileHandleRegister)(CUfileHandle_t *, CUfileDescr_t *) = nullptr;
  CUfileError_t (*cuFileHandleDeregister)(CUfileHandle_t) = nullptr;
  ssize_t (*cuFileRead)(CUfileHandle_t, void *, size_t, off_t, size_t) = nullptr;
  ssize_t (*cuFileWrite)(CUfileHandle_t, const void *, size_t, off_t, size_t) = nullptr;
  CUfileError_t (*cuFileBufRegister)(void *, size_t, uint32_t) = nullptr;
  CUfileError_t (*cuFileBufDeregister)(void *) = nullptr;

  CuFileApi() {
    void *handle = dlopen("libcufile.so", RTLD_LAZY);
    if (!handle) {
      std::cerr << "Failed to load libcufile.so: " << dlerror() << std::endl;
      exit(EXIT_FAILURE);
    }

    cuFileHandleRegister = (CUfileError_t (*)(CUfileHandle_t *, CUfileDescr_t *))dlsym(handle, "cuFileHandleRegister");
    cuFileHandleDeregister = (CUfileError_t (*)(CUfileHandle_t))dlsym(handle, "cuFileHandleDeregister");
    cuFileRead = (ssize_t (*)(CUfileHandle_t, void *, size_t, off_t, size_t))dlsym(handle, "cuFileRead");
    cuFileWrite = (ssize_t (*)(CUfileHandle_t, const void *, size_t, off_t, size_t))dlsym(handle, "cuFileWrite");
    cuFileBufRegister = (CUfileError_t (*)(void *, size_t, uint32_t))dlsym(handle, "cuFileBufRegister");
    cuFileBufDeregister = (CUfileError_t (*)(void *))dlsym(handle, "cuFileBufDeregister");

    if (!cuFileHandleRegister || !cuFileHandleDeregister || !cuFileRead || 
        !cuFileWrite || !cuFileBufRegister || !cuFileBufDeregister) {
      std::cerr << "Failed to load cuFile symbols: " << dlerror() << std::endl;
      exit(EXIT_FAILURE);
    }
  }
};

/** Pointers to the real stdio API */
class StdioApi : public RealApi {
 public:

  typedef FILE *(*fopen_t)(const char *path, const char *mode);
  typedef FILE *(*fopen64_t)(const char *path, const char *mode);
  /** fopen */
  fopen_t fopen = nullptr;
  /** fopen64 */
  fopen64_t fopen64 = nullptr;

  StdioApi() : RealApi("fopen", "stdio_intercepted") {
    fopen = (fopen_t)dlsym(real_lib_, "fopen");
    REQUIRE_API(fopen)
    fopen64 = (fopen64_t)dlsym(real_lib_, "fopen64");
    REQUIRE_API(fopen64)
  }
};

}  // namespace hermes::adapter

#include "hermes_shm/util/singleton.h"

// Singleton macros
#define HERMES_CUFILE_API \
  hshm::EasySingleton<::hermes::adapter::CuFileApi>::GetInstance()
#define HERMES_STDIO_API \
  hshm::EasySingleton<::hermes::adapter::StdioApi>::GetInstance()
#define HERMES_STDIO_API_T hermes::adapter::StdioApi*

#endif  // HERMES_ADAPTER_STDIO_H
