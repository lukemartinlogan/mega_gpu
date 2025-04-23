#include <assert.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>

#include "hermes_shm/hermes_shm.h"
#include "so_test.h"

int main() {
  SharedTest::run();
  auto mem_mngr = HSHM_MEMORY_MANAGER;

  MemoryManager2 mngr;
  mngr.function<0>();
  mem_mngr->ScanBackendsGpu<1>(0);
  mem_mngr->CreateBackend<hipc::PosixShmMmap>(hipc::MemoryBackendId::Get(0),
                                              MEGABYTES(10), "cmalsdfs");
  mem_mngr->CreateAllocator<hipc::ThreadLocalAllocator>(
      hipc::MemoryBackendId::Get(0), hipc::AllocatorId(1, 0), 0);
  return 0;
}
