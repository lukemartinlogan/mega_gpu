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
#include "hermes_shm/util/logging.h"
#include <cstdio>
#include "hermes_adapters/real_api.h"

extern "C" {
typedef FILE * (*fopen_t)(const char * path, const char * mode);
typedef FILE * (*fopen64_t)(const char * path, const char * mode);
}

namespace hermes::adapter {

/** Pointers to the real stdio API */
class StdioApi : public RealApi {
 public:
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
#define HERMES_STDIO_API \
  hshm::EasySingleton<::hermes::adapter::StdioApi>::GetInstance()
#define HERMES_STDIO_API_T hermes::adapter::StdioApi*

#endif  // HERMES_ADAPTER_STDIO_H
