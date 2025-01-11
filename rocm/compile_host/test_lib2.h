#ifndef TEST_LIB2_H
#define TEST_LIB2_H

#include "mycommon.h"

/**
 * A class to represent singleton pattern
 * Does not require specific initialization of the static variable
 * */
template <typename T, bool WithLock>
class SingletonBase {
 public:
  HOST_DEV
  static T *GetInstance() {
    if (GetObject() == nullptr) {
      new ((T *)GetData()) T();
      GetObject() = (T *)GetData();
    }
    return GetObject();
  }

  HOST_DEV
  static T *GetData() {
    static char data_[sizeof(T)] = {0};
    return (T *)data_;
  }

  HOST_DEV
  static T *&GetObject() {
    static T *obj_ = nullptr;
    return obj_;
  }
};

/** Singleton default case declaration */
template <typename T>
using Singleton = SingletonBase<T, true>;

/** Singleton without lock declaration */
template <typename T>
using LockfreeSingleton = SingletonBase<T, false>;

class TestLib2 {
 public:
  HOST_DEV size_t GetSize();

  HOST_DEV void GetSize2();
};

#endif  // TEST_LIB2_H