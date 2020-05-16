#include "InferenceEngine.hpp"
#include <fstream>

#include <NvInfer.h>

namespace
{
  // From TensorRT example : 'sample_uff_mnist'

  inline unsigned int elementSize(nvinfer1::DataType t)
  {
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
      // Fallthrough, same as kFLOAT
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    }
    assert(0);
    return 0;
  }

  bool writeBufferToFile(void* buffer, size_t size, const char* path)
  {
    if (!buffer)
      return false;

    std::ofstream stream(path, std::ios::binary);
    if (!stream)
      return false;

    stream.write(static_cast<char*>(buffer), size);
    return true;
  }

  std::string readBufferFromFile(const char* path)
  {
    std::string buffer;
    std::ifstream stream(path, std::ios::binary);

    if (stream)
    {
      stream >> std::noskipws;
      std::copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
    }

    return buffer;
  }

  inline uint64_t volume(const nvinfer1::Dims& d)
  {
    uint64_t v = 1;
    for (uint64_t i = 0; i < d.nbDims; i++)
      v *= d.d[i];
    return v;
  };

  inline size_t bytesToMB(size_t bytes)
  {
    return static_cast<size_t> (bytes / (1024.f * 1024.f));
  }
}


InferenceEngine::InferenceEngine(const char* name, int batchSize) : mName(name), mBatchSize(batchSize)
{
}

InferenceEngine::~InferenceEngine()
{
}

InferenceEngine::UniquePtr InferenceEngine::create(const std::string& path, const char* name, int batchSize)
{
  InferenceEngine::UniquePtr pInferenceEngine = std::make_unique<InferenceEngine>(name, batchSize);
  if (!pInferenceEngine->deserialze(path.c_str()))
    return nullptr;
  if (!pInferenceEngine->createExecutionContext())
    return nullptr;
  pInferenceEngine->calculateDeviceMemoryinMB();
  return pInferenceEngine;
}

InferenceEngine::UniquePtr InferenceEngine::create(nvinfer1::ICudaEngine* pEngine, const char* name, int batchSize)
{
  if (!pEngine) return nullptr;
  InferenceEngine::UniquePtr pInferenceEngine = std::make_unique<InferenceEngine>(name, batchSize);
  pInferenceEngine->mpEngine.reset(pEngine);
  if (!pInferenceEngine->createExecutionContext()) return nullptr;
  pInferenceEngine->calculateDeviceMemoryinMB();
  return pInferenceEngine;
}

bool InferenceEngine::serialze(const std::string& path) const
{
  if (!mpEngine) return false;
  auto hostMemory = mpEngine->serialize();
  bool success = writeBufferToFile(hostMemory->data(), hostMemory->size(), path.c_str());
  hostMemory->destroy();
  return success;
}

bool InferenceEngine::deserialze(const std::string& path)
{
  Logger logger;
  std::string buffer = readBufferFromFile(path.c_str());
  std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter> runtime{ nvinfer1::createInferRuntime(logger) };
  if (buffer.empty())
    return false;
  mpEngine = std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>(runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr));
  return mpEngine ? true : false;
}

void InferenceEngine::reset()
{
  mpEngine.reset();
  mpContext.reset();
}

void InferenceEngine::calculateDeviceMemoryinMB()
{
  mEngineDeviceMemory = static_cast<size_t> (mpEngine ? bytesToMB(mpEngine->getDeviceMemorySize()) : 0.f);
}

bool InferenceEngine::createExecutionContext()
{
  mWorkspaceSize = mpEngine->getWorkspaceSize();
  mpContext = std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter>(mpEngine->createExecutionContext());
  return mpContext ? true : false;
}

std::vector<InferenceEngine::BufferInfo> InferenceEngine::getBufferInfos() const
{
  if (!mpEngine)
    return {};

  std::vector<BufferInfo> bufferInfos;
  for (int i = 0; i < mpEngine->getNbBindings(); i++)
  {
      nvinfer1::Dims dims = mpEngine->getBindingDimensions(i);
    BufferInfo info;
    info.mName = mpEngine->getBindingName(i);
    info.mDtype = mpEngine->getBindingDataType(i);
    info.mSize = volume(dims) * elementSize(info.mDtype) * mBatchSize;
    info.mIsInput = mpEngine->bindingIsInput(i);
    bufferInfos.push_back(info);
  }
  return std::move(bufferInfos);
}

std::vector<CudaBuffer<void>> InferenceEngine::generateCudaBuffers(bool generateInputBuffers)
{
  auto bufferInfos = getBufferInfos();
  mAllocatedBufferMemory = 0;

  std::vector<CudaBuffer<void>> cudaBuffers;
  for (const auto& bufferInfo : bufferInfos)
  {
    if (!bufferInfo.mIsInput || generateInputBuffers)
    {
      mAllocatedBufferMemory += bufferInfo.mSize;
      cudaBuffers.emplace_back(CudaBuffer<void>::create(bufferInfo.mSize));
    }
  }
  mAllocatedBufferMemory = bytesToMB(mAllocatedBufferMemory);
  return std::move(cudaBuffers);
}

void InferenceEngine::executeAsync(void** bindings, FalcorCUDA::cudaStream_t stream)
{
  mpContext->enqueue(mBatchSize, bindings, (cudaStream_t) stream, nullptr);
}

void InferenceEngine::executeBlocking(void** bindings)
{
  mpContext->execute(mBatchSize, bindings);
}

int InferenceEngine::getNumInputs() const
{
  auto numBindings = mpEngine->getNbBindings();
  int numInputs = 0;
  for (int binding = 0; binding < numBindings; binding++)
    if (mpEngine->bindingIsInput(binding)) numInputs++;
  return numInputs;
}

int InferenceEngine::getNumOutputs() const
{
  auto numBindings = mpEngine->getNbBindings();
  int numOutputs = 0;
  for (int binding = 0; binding < numBindings; binding++)
    if (!mpEngine->bindingIsInput(binding)) numOutputs++;
  return numOutputs;
}
