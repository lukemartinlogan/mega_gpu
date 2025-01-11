
#include "test_lib.h"

#include "mycommon.h"
#include "test_lib2.h"

HOST_DEV size_t TestLib::GetSize() {
  int sz = Singleton<TestLib2>::GetInstance()->GetSize();
  return 1 + sz;
}

HOST_DEV void TestLib::GetSize2() {
  Singleton<TestLib2>::GetInstance()->GetSize();
}