#pragma once

#include "Falcor.h"
#include "FalcorCUDA.h"
#include "Utils/Cuda/CudaBuffer.h"

#include "NvInfer.h"

#include <memory>
#include <string>
#include <vector>


// Logger for TensorRT info/warning/errors (TensorRT example : 'sample_uff_mnist')
class Logger : public nvinfer1::ILogger
{
public:
  Logger(Severity severity = Severity::kERROR) : reportableSeverity(severity) {}

  void log(Severity severity, const char* msg) override
  {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity)
      return;

    switch (severity)
    {
    case Severity::kINTERNAL_ERROR: Falcor::logError(msg);
        break;
    case Severity::kERROR: Falcor::logError(msg);
        break;
    case Severity::kWARNING: Falcor::logWarning(msg);
        break;
    case Severity::kINFO: Falcor::logInfo(msg);
        break;
    default: break;
    }
  }

  Severity reportableSeverity;
};


namespace nvinfer1
{
    class ICudaEngine;
    class IExecutionContext;
}

class InferenceEngine
{
  public:
    using UniquePtr = std::unique_ptr<InferenceEngine>;

    struct NvInferDeleter
    {
      template <typename T>
      void operator()(T* obj) const
      {
        if (obj)
        {
          obj->destroy();
        }
      }
    };

    struct BufferInfo
    {
      std::string mName;
      nvinfer1::DataType mDtype;
      uint64_t mSize = 0;
      bool mIsInput = false;
    };

    static InferenceEngine::UniquePtr create(const std::string& path, const char* name,  int batchSize = 1);
    static InferenceEngine::UniquePtr create(nvinfer1::ICudaEngine* pEngine, const char* name, int batchSize = 1);

    InferenceEngine(const char* name, int batchSize);
    ~InferenceEngine();

    bool isValid() const { return mpEngine != nullptr && mpContext != nullptr; }
    bool serialze(const std::string& path) const;
    void reset();

    std::vector<BufferInfo> getBufferInfos() const;
    std::vector<CudaBuffer<void>> generateCudaBuffers(bool generateInputBuffers);

    size_t getEngineDeviceMemoryInMB() const { return mEngineDeviceMemory; }
    size_t getAllocatedBufferMemoryMB() const { return mAllocatedBufferMemory; }
    nvinfer1::ICudaEngine* getEngine() const { return mpEngine.get(); }
    nvinfer1::IExecutionContext* getExecutionContext() const { return mpContext.get(); }
    const std::string& getName() const { return mName; }

    int getNumInputs() const;
    int getNumOutputs() const;

    void executeAsync(void** bindings, FalcorCUDA::cudaStream_t stream);
    void executeBlocking(void** bindings);

  private:
    void calculateDeviceMemoryinMB();
    bool createExecutionContext();
    bool deserialze(const std::string& path);

    std::string mName;
    std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter> mpEngine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter> mpContext = nullptr;
    int mBatchSize;

    size_t mWorkspaceSize = 0;
    size_t mEngineDeviceMemory = 0;
    size_t mAllocatedBufferMemory = 0;
};
