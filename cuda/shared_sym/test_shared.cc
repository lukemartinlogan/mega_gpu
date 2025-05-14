#include <assert.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <thread>

#include "hermes_shm/hermes_shm.h"
#include "so_test.h"

int main() {
  void *handle = dlopen("libsimple_lib_so.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load library: " << dlerror() << std::endl;
    return 1;
  }
  void (*run_func)() = (void (*)())dlsym(handle, "SharedTestRun");
  if (!run_func) {
    std::cerr << "Failed to load symbol: " << dlerror() << std::endl;
    dlclose(handle);
    return 1;
  }
  run_func();
  std::thread worker_thread(run_func);
  worker_thread.join();
  return 0;
}
