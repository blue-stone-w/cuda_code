#ifndef REDUCTION_H
#define REDUCTION_H

#include "common/cuda_base.h"

#include <iostream>


#ifdef __cplusplus
extern "C" bool adds(int *idata_host, int *odata_host, int size, int blocksize);

#endif


class Reduction
{
 public:
  Reduction( );
  ~Reduction( );

  int size = 1 << 24;
  size_t bytes;
  int *idata_host;
  int *odata_host;
  int *tmp;
  int gpu_sum = 0;
  void initialData_int(int *ip, int size);
};

#endif