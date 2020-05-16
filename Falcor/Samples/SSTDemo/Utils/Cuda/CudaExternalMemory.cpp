#include "CudaExternalMemory.h"
#include <comip.h>
#include <utility>


CudaExternalMemory::CudaExternalMemory(CudaExternalMemory&& other)
{
  if (this->mExternalMemory) this->free();
  mpCudaDevMemory = other.mpCudaDevMemory;
  mExternalMemory = other.mExternalMemory;
  mpBuffer = std::move(other.mpBuffer);
  mSize = other.mSize;
  other.mpCudaDevMemory = nullptr;
  other.mExternalMemory = nullptr;
  other.mSize = 0;
}

CudaExternalMemory& CudaExternalMemory::operator=(CudaExternalMemory&& other)
{
  if (this->mExternalMemory) this->free();
  this->mpCudaDevMemory = other.mpCudaDevMemory;
  this->mExternalMemory = other.mExternalMemory;
  this->mpBuffer = std::move(other.mpBuffer);
  this->mSize = other.mSize;
  other.mpCudaDevMemory = nullptr;
  other.mExternalMemory = nullptr;
  other.mSize = 0;
  return *this;
}

void CudaExternalMemory::free()
{
  unmapResource(mExternalMemory);
  Falcor::Cuda::free(mpCudaDevMemory);
}

CudaExternalMemory::~CudaExternalMemory()
{
  free();
}

bool CudaExternalMemory::memset(int value) const
{
    if (mpCudaDevMemory)
        Falcor::Cuda::memset(mpCudaDevMemory, value, mSize);
    return mpCudaDevMemory ? true : false;
}

bool CudaExternalMemory::readToHost(void* dst, size_t size) const
{
    if (mpCudaDevMemory)
        Falcor::Cuda::memcpy(dst, mpCudaDevMemory, (size > 0 ? size : mSize), FalcorCUDA::cudaMemcpyDeviceToHost);
    return mpCudaDevMemory ? true : false;
}

bool CudaExternalMemory::writeToDevice(void* src, size_t size) const
{
    if (mpCudaDevMemory)
        Falcor::Cuda::memcpy(mpCudaDevMemory, src, (size > 0 ? size : mSize), FalcorCUDA::cudaMemcpyHostToDevice);
    return mpCudaDevMemory ? true : false;
}

CudaExternalMemory CudaExternalMemory::create(Falcor::Buffer::SharedPtr buffer)
{
  CudaExternalMemory mappedMemory;
  mappedMemory.mpBuffer = buffer;
  mappedMemory.mSize = buffer->getSize();
  mapResource(buffer.get(), mappedMemory.mSize, mappedMemory.mExternalMemory, &mappedMemory.mpCudaDevMemory);
  return std::move(mappedMemory);
}
