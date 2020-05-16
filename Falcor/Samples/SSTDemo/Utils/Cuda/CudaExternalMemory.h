#pragma once

#include "Falcor.h"
#include "FalcorCUDA.h"
#include <cstddef>
#include <utility>

class CudaExternalMemory
{
public:
  CudaExternalMemory() = default;
  CudaExternalMemory(CudaExternalMemory&& other);
  CudaExternalMemory& operator=(CudaExternalMemory&& other);
  ~CudaExternalMemory();

  void* data() const { return mpCudaDevMemory; }
  bool memset(int value) const;
  bool readToHost(void* dst, size_t size = 0) const;
  bool writeToDevice(void* src, size_t size = 0) const;
  bool isMapped(const Falcor::Buffer::SharedPtr& buffer) const { return mpCudaDevMemory != nullptr && buffer == mpBuffer; }
  size_t size() const { return mSize; }
  operator void*() const { return mpCudaDevMemory; }

  static CudaExternalMemory create(Falcor::Buffer::SharedPtr pBuffer);

  CudaExternalMemory(const CudaExternalMemory&) = delete;
  CudaExternalMemory& operator=(const CudaExternalMemory&) = delete;

private:
  void free();

  Falcor::Buffer::SharedPtr mpBuffer;
  void* mpCudaDevMemory = nullptr;
  FalcorCUDA::cudaExternalMemory_t mExternalMemory = nullptr;
  size_t mSize = 0;
};
