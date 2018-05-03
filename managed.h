#ifndef PATCHMATCH_MANAGED_H
#define PATCHMATCH_MANAGED_H

#include <cuda_runtime_api.h>
#include "helper_cuda.h"

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, len));
    return ptr;
  }

  void operator delete(void *ptr) {
      cudaFree(ptr);
  }
};

#endif