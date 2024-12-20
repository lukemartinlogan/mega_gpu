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

bool stdio_intercepted = true;

#include <limits.h>
#include <sys/file.h>
#include <cstdio>
#include "stdio_api.h"
#include "stdio_fs_api.h"
#include "cufile_api.h"


using hermes::adapter::MetadataManager;
using hermes::adapter::SeekMode;
using hermes::adapter::AdapterStat;
using hermes::adapter::File;
using hermes::adapter::IoStatus;
using hermes::adapter::FsIoOptions;

namespace stdfs = std::filesystem;

extern "C" {
/**
 * STDIO
 */
// Intercept cuFileOpen
int cuFileOpen(int *fd, const char *path, int flags) {
  std::cout << "Intercepted cuFileOpen: " << path << std::endl;
  return HERMES_CUFILE_API->cuFileOpen(fd, path, flags);
}

// Intercept cuFileRead
ssize_t cuFileRead(void *devPtr, size_t size, off_t fileOffset, int fd) {
  std::cout << "Intercepted cuFileRead: " << size << " bytes" << std::endl;
  return HERMES_CUFILE_API->cuFileRead(devPtr, size, fileOffset, fd);
}

// Intercept cuFileWrite
ssize_t cuFileWrite(const void *devPtr, size_t size, off_t fileOffset, int fd) {
  std::cout << "Intercepted cuFileWrite: " << size << " bytes" << std::endl;
  return HERMES_CUFILE_API->cuFileWrite(devPtr, size, fileOffset, fd);
}

// Intercept cuFileClose
int cuFileClose(int fd) {
  std::cout << "Intercepted cuFileClose: " << fd << std::endl;
  return HERMES_CUFILE_API->cuFileClose(fd);
}
FILE *HERMES_DECL(fopen)(const char *path, const char *mode) {
  TRANSPARENT_HERMES();
  auto real_api = HERMES_STDIO_API;
  auto fs_api = HERMES_STDIO_FS;
  if (fs_api->IsPathTracked(path)) {
    HILOG(kDebug, "Intercepting fopen({}, {})", path, mode)
    AdapterStat stat;
    stat.mode_str_ = mode;
    return fs_api->Open(stat, path).hermes_fh_;
  } else {
    return real_api->fopen(path, mode);
  }
}


/**
 * STDIO
 */

FILE *HERMES_DECL(fopen)(const char *path, const char *mode) {
  TRANSPARENT_HERMES();
  auto real_api = HERMES_STDIO_API;
  auto fs_api = HERMES_STDIO_FS;
  if (fs_api->IsPathTracked(path)) {
    HILOG(kDebug, "Intercepting fopen({}, {})", path, mode)
    AdapterStat stat;
    stat.mode_str_ = mode;
    return fs_api->Open(stat, path).hermes_fh_;
  } else {
    return real_api->fopen(path, mode);
  }
}

FILE *HERMES_DECL(fopen64)(const char *path, const char *mode) {
  TRANSPARENT_HERMES();
  auto real_api = HERMES_STDIO_API;
  auto fs_api = HERMES_STDIO_FS;
  if (fs_api->IsPathTracked(path)) {
    HILOG(kDebug, "Intercepting fopen64({}, {})", path, mode)
    AdapterStat stat;
    stat.mode_str_ = mode;
    return fs_api->Open(stat, path).hermes_fh_;
  } else {
    return real_api->fopen64(path, mode);
  }
}

}  // extern C
