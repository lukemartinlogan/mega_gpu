
#include "test_lib2.h"

#include "test_lib.h"

HOST_DEV size_t TestLib2::GetSize() {
#ifdef HSHM_IS_HOST
  return 0;
#else
  return 1;
#endif
}

HOST_DEV void TestLib2::GetSize2() {
#ifdef HSHM_IS_HOST
  TestLib test_lib;
  test_lib.GetSize2();
  return;
#else
  return;
#endif
}