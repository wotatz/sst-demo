#pragma once

#include "FalcorCUDA.h"

class CuEventTimer
{
private:
  FalcorCUDA::cudaEvent_t eventStart;
  FalcorCUDA::cudaEvent_t eventEnd;
  FalcorCUDA::cudaStream_t stream;

public:
  CuEventTimer(FalcorCUDA::cudaStream_t stream = 0) : stream(stream)
  {
    checkCudaError(FalcorCUDA::cudaEventCreate(&eventStart));
    checkCudaError(FalcorCUDA::cudaEventCreate(&eventEnd));
  }

  ~CuEventTimer()
  {
      checkCudaError(FalcorCUDA::cudaEventDestroy(eventStart));
      checkCudaError(FalcorCUDA::cudaEventDestroy(eventEnd));
  }

  void setStream(FalcorCUDA::cudaStream_t stream)
  {
      this->stream = stream;
  }

  void start()
  {
    checkCudaError(FalcorCUDA::cudaEventRecord(eventStart, stream));
  }

  void end()
  {
    checkCudaError(FalcorCUDA::cudaEventRecord(eventEnd, stream));
  }

  float getElapsedTime()
  {
    float time;
    checkCudaError(FalcorCUDA::cudaEventElapsedTime(&time, eventStart, eventEnd));
    return time;
  }

  void sync()
  {
    checkCudaError(FalcorCUDA::cudaEventSynchronize(eventEnd));
  }
};
