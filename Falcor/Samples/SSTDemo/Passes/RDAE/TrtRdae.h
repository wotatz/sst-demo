#pragma once

#include "Falcor.h"
#include "FalcorCUDA.h"

#include "Utils/Cuda/CudaBuffer.h"
#include "Utils/Cuda/CudaExternalMemory.h"
#include "Utils/TRT/InferenceEngine.hpp"

#include <memory>
#include <string>
#include <vector>

class Rdae;

class TrtRdae
{
  friend Rdae;

public:
  /** Create inference engine. */
  void create(ivec2 size, bool fp16);

  /** Get Tensor-RT inference engine. */
  const auto& getInferenceEngine() const { return mpFilterEngine; }

  /** Execute inference. */
  bool infer(CudaExternalMemory& inColor, CudaExternalMemory& inAux, CudaExternalMemory& outColor, bool clearRecurrentBuffers);

  /** Get CUDA stream. */
  FalcorCUDA::cudaStream_t& getCudaStream() { return mStream; }

  TrtRdae();
  ~TrtRdae();

private:
  size_t setupCudaBuffers();
  void setupRawBuffers();

  bool mFp16 = true;

  // Memory info
  size_t mTotalDeviceMemory = 0;
  size_t mFreeDeviceMemory = 0;
  size_t mTotalEngineDeviceMemory = 0;
  size_t mTotalBufferDeviceMemory = 0;
 
  // Inference Engine
  InferenceEngine::UniquePtr mpInputEngine;
  InferenceEngine::UniquePtr mpFilterEngine;

  // Cuda buffers needed for inference
  CudaBuffer<void> mBufferInputImage;
  CudaBuffer<void> mBufferInputAux;

  std::vector<CudaBuffer<void>> mBuffersRecurrent;

  // Raw cuda pointer buffers for inference calls
  std::vector<void*> mRawInputBuffers;
  std::vector<void*> mRawFilterBuffers;

  FalcorCUDA::cudaStream_t mStream;
};
