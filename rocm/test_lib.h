#include <cstddef>

#include "hermes_shm/constants/macros.h"

class TestLib {
 public:
  HSHM_DLL HSHM_CROSS_FUN size_t GetSize();
};