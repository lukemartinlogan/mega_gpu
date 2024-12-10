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

#ifndef HERMES_ADAPTER_POSIX_H
#define HERMES_ADAPTER_POSIX_H
#include <string>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include "real_api.h"
#include <hermes_shm/util/singleton.h>

#ifndef O_TMPFILE
#define O_TMPFILE 0
#endif

#ifndef _STAT_VER
#define _STAT_VER 0
#endif

extern "C" {
typedef int (*open_t)(const char * path, int flags,  ...);
typedef int (*open64_t)(const char * path, int flags,  ...);
}

namespace hermes::adapter {

/** Used for compatability with older kernel versions */
static int fxstat_to_fstat(int fd, struct stat * stbuf);

/** Pointers to the real posix API */
class PosixApi : public RealApi {
 public:
  bool is_loaded_ = false;

 public:
  /** open */
  open_t open = nullptr;
  /** open64 */
  open64_t open64 = nullptr;

  PosixApi() : RealApi("open", "posix_intercepted") {
    open = (open_t)dlsym(real_lib_, "open");
    REQUIRE_API(open)
    open64 = (open64_t)dlsym(real_lib_, "open64");
    REQUIRE_API(open64)
  }

  bool IsInterceptorLoaded() {
    if (is_loaded_) {
      return true;
    }
    InterceptorApi<PosixApi> check("open", "posix_intercepted");
    is_loaded_ = check.is_loaded_;
    return is_loaded_;
  }
};

}  // namespace hermes::adapter

// Singleton macros
#include "hermes_shm/util/singleton.h"

#define HERMES_POSIX_API \
  hshm::EasySingleton<::hermes::adapter::PosixApi>::GetInstance()
#define HERMES_POSIX_API_T hermes::adapter::PosixApi*


namespace hermes::adapter {
/** Used for compatability with older kernel versions */
static int fxstat_to_fstat(int fd, struct stat *stbuf) {
#ifdef _STAT_VER
  return HERMES_POSIX_API->__fxstat(_STAT_VER, fd, stbuf);
#endif
}
}  // namespace hermes::adapter

#endif  // HERMES_ADAPTER_POSIX_H
