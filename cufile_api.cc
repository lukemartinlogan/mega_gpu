#include "cufile_api.h"

extern "C" {

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

}
