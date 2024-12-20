

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

extern "C"
{
  typedef CUfileError_t (*cuFileHandleRegister_t)(CUfileHandle_t *, CUfileDescr_t *);
  typedef void (*cuFileHandleDeregister_t)(CUfileHandle_t);
  typedef CUfileError_t (*cuFileBufRegister_t)(const void *, size_t, int);
  typedef CUfileError_t (*cuFileBufDeregister_t)(const void *);
  typedef ssize_t (*cuFileRead_t)(CUfileHandle_t, void *, size_t, off_t, off_t);
  typedef ssize_t (*cuFileWrite_t)(CUfileHandle_t, const void *, size_t, off_t, off_t);
  typedef CUfileError_t (*cuFileDriverOpen_t)(void);
  typedef CUfileError_t (*cuFileDriverClose_t)(void);
  typedef long (*cuFileUseCount_t)(void);
  typedef CUfileError_t (*cuFileDriverGetProperties_t)(CUfileDrvProps_t *);
  typedef CUfileError_t (*cuFileDriverSetPollMode_t)(bool, size_t);
  typedef CUfileError_t (*cuFileDriverSetMaxDirectIOSize_t)(size_t);
  typedef CUfileError_t (*cuFileDriverSetMaxCacheSize_t)(size_t);
  typedef CUfileError_t (*cuFileDriverSetMaxPinnedMemSize_t)(size_t);
  typedef CUfileError_t (*cuFileBatchIOSetUp_t)(CUfileBatchHandle_t *, unsigned);
  typedef CUfileError_t (*cuFileBatchIOSubmit_t)(CUfileBatchHandle_t, unsigned, CUfileIOParams_t *, unsigned int);
  typedef CUfileError_t (*cuFileBatchIOGetStatus_t)(CUfileBatchHandle_t, unsigned, unsigned *, CUfileIOEvents_t *, struct timespec *);
  typedef CUfileError_t (*cuFileBatchIOCancel_t)(CUfileBatchHandle_t);
  typedef void (*cuFileBatchIODestroy_t)(CUfileBatchHandle_t);
}

namespace hermes::adapter
{

  /** CuFileApi class */
  class CuFileApi : public RealApi
  {
  public:
    cuFileHandleRegister_t cuFileHandleRegister = nullptr;
    cuFileHandleDeregister_t cuFileHandleDeregister = nullptr;
    cuFileBufRegister_t cuFileBufRegister = nullptr;
    cuFileBufDeregister_t cuFileBufDeregister = nullptr;
    cuFileRead_t cuFileRead = nullptr;
    cuFileWrite_t cuFileWrite = nullptr;
    cuFileDriverOpen_t cuFileDriverOpen = nullptr;
    cuFileDriverClose_t cuFileDriverClose = nullptr;
    cuFileUseCount_t cuFileUseCount = nullptr;
    cuFileDriverGetProperties_t cuFileDriverGetProperties = nullptr;
    cuFileDriverSetPollMode_t cuFileDriverSetPollMode = nullptr;
    cuFileDriverSetMaxDirectIOSize_t cuFileDriverSetMaxDirectIOSize = nullptr;
    cuFileDriverSetMaxCacheSize_t cuFileDriverSetMaxCacheSize = nullptr;
    cuFileDriverSetMaxPinnedMemSize_t cuFileDriverSetMaxPinnedMemSize = nullptr;
    cuFileBatchIOSetUp_t cuFileBatchIOSetUp = nullptr;
    cuFileBatchIOSubmit_t cuFileBatchIOSubmit = nullptr;
    cuFileBatchIOGetStatus_t cuFileBatchIOGetStatus = nullptr;
    cuFileBatchIOCancel_t cuFileBatchIOCancel = nullptr;
    cuFileBatchIODestroy_t cuFileBatchIODestroy = nullptr;

    CuFileApi() : RealApi("cuFileHandleRegister", "cuFile_Intercepted")
    {

      cuFileHandleRegister = (cuFileHandleRegister_t)dlsym(real_lib_, "cuFileHandleRegister");
      cuFileHandleDeregister = (cuFileHandleDeregister_t)dlsym(real_lib_, "cuFileHandleDeregister");
      cuFileBufRegister = (cuFileBufRegister_t)dlsym(real_lib_, "cuFileBufRegister");
      cuFileBufDeregister = (cuFileBufDeregister_t)dlsym(real_lib_, "cuFileBufDeregister");
      cuFileRead = (cuFileRead_t)dlsym(real_lib_, "cuFileRead");
      cuFileWrite = (cuFileWrite_t)dlsym(real_lib_, "cuFileWrite");
      cuFileDriverOpen = (cuFileDriverOpen_t)dlsym(real_lib_, "cuFileDriverOpen");
      cuFileDriverClose = (cuFileDriverClose_t)dlsym(real_lib_, "cuFileDriverClose");
      cuFileUseCount = (cuFileUseCount_t)dlsym(real_lib_, "cuFileUseCount");
      cuFileDriverGetProperties = (cuFileDriverGetProperties_t)dlsym(real_lib_, "cuFileDriverGetProperties");
      cuFileDriverSetPollMode = (cuFileDriverSetPollMode_t)dlsym(real_lib_, "cuFileDriverSetPollMode");
      cuFileDriverSetMaxDirectIOSize = (cuFileDriverSetMaxDirectIOSize_t)dlsym(real_lib_, "cuFileDriverSetMaxDirectIOSize");
      cuFileDriverSetMaxCacheSize = (cuFileDriverSetMaxCacheSize_t)dlsym(real_lib_, "cuFileDriverSetMaxCacheSize");
      cuFileDriverSetMaxPinnedMemSize = (cuFileDriverSetMaxPinnedMemSize_t)dlsym(real_lib_, "cuFileDriverSetMaxPinnedMemSize");
      cuFileBatchIOSetUp = (cuFileBatchIOSetUp_t)dlsym(real_lib_, "cuFileBatchIOSetUp");
      cuFileBatchIOSubmit = (cuFileBatchIOSubmit_t)dlsym(real_lib_, "cuFileBatchIOSubmit");
      cuFileBatchIOGetStatus = (cuFileBatchIOGetStatus_t)dlsym(real_lib_, "cuFileBatchIOGetStatus");
      cuFileBatchIOCancel = (cuFileBatchIOCancel_t)dlsym(real_lib_, "cuFileBatchIOCancel");
      cuFileBatchIODestroy = (cuFileBatchIODestroy_t)dlsym(real_lib_, "cuFileBatchIODestroy");

      if (!cuFileHandleRegister || !cuFileHandleDeregister || !cuFileRead ||
          !cuFileWrite || !cuFileBufRegister || !cuFileBufDeregister)
      {
        std::cerr << "Failed to load cuFile symbols: " << dlerror() << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  };

  /** Pointers to the real stdio API */
  class StdioApi : public RealApi
  {
  public:
    typedef FILE *(*fopen_t)(const char *path, const char *mode);
    typedef FILE *(*fopen64_t)(const char *path, const char *mode);
    /** fopen */
    fopen_t fopen = nullptr;
    /** fopen64 */
    fopen64_t fopen64 = nullptr;

    StdioApi() : RealApi("fopen", "stdio_intercepted")
    {
      fopen = (fopen_t)dlsym(real_lib_, "fopen");
      REQUIRE_API(fopen)
      fopen64 = (fopen64_t)dlsym(real_lib_, "fopen64");
      REQUIRE_API(fopen64)
    }
  };

} // namespace hermes::adapter

#include "hermes_shm/util/singleton.h"

// Singleton macros
#define HERMES_CUFILE_API \
  hshm::EasySingleton<::hermes::adapter::CuFileApi>::GetInstance()
#define HERMES_STDIO_API \
  hshm::EasySingleton<::hermes::adapter::StdioApi>::GetInstance()
#define HERMES_STDIO_API_T hermes::adapter::StdioApi *

#endif // HERMES_ADAPTER_STDIO_H
