//
// Created by llogan on 10/18/24.
//

#ifndef MEGAGPU__SIMPLE_LIB_H_
#define MEGAGPU__SIMPLE_LIB_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "hermes_shm/hermes_shm.h"

extern "C" {
void SharedTestRun();
}

#endif  // MEGAGPU__SIMPLE_LIB_H_
