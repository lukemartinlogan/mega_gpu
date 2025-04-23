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

#include "so_test.h"

int main() {
  printf("HERE!!!\n");
  MemoryManager2 mngr;
  mngr.function<0>();

  return 0;
}