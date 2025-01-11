#ifndef TEST_LIB_H
#define TEST_LIB_H

#include "mycommon.h"

class TestLib {
 public:
  HOST_DEV size_t GetSize();

  HOST_DEV void GetSize2();
};

#endif  // TEST_LIB_H