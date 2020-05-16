#pragma once

#include "FalcorCUDA.h"
#include <cstddef>
#include <string>

template <typename T=void>
class CudaBuffer
{
public:
  CudaBuffer() = default;

  CudaBuffer(CudaBuffer&& other)
  {
    mBuffer = other.mBuffer;
    mSize = other.mSize;
    other.mBuffer = nullptr;
    other.mSize = 0;
  }

  CudaBuffer& operator=(CudaBuffer&& other)
  {
    this->mBuffer = other.mBuffer;
    this->mSize = other.mSize;
    other.mBuffer = nullptr;
    other.mSize = 0;
    return *this;
  }

  ~CudaBuffer()
  {
    Falcor::Cuda::free(mBuffer);
  }

  void* data()      const { return mBuffer; }
  T* dataAsType()   const { return static_cast<T*> (mBuffer); }
  size_t size()     const { return mSize; }
  operator void*()  const { return mBuffer; }

  void memset(int value)
  {
    Falcor::Cuda::memset(mBuffer, value, mSize);
  }

  void memset(int value, size_t count)
  {
    Falcor::Cuda::memset(mBuffer, value, count);
  }

  static CudaBuffer<T> create(size_t size_in_bytes)
  {
    CudaBuffer<T> cuBuffer;
    cuBuffer.mBuffer = Falcor::Cuda::malloc(size_in_bytes);
    cuBuffer.mSize = size_in_bytes;
    return std::move(cuBuffer);
  }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

private:
    void* mBuffer{ nullptr };
    size_t mSize{ 0 };
};
