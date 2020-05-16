#include "TrtRdae.h"
#include "FalcorCUDA.h"

#include <NvInfer.h>
#include <NvUffParser.h>

using namespace FalcorCUDA;

#include <fstream>
#include <functional>

static Logger gLogger(nvinfer1::ILogger::Severity::kINFO);

namespace
{
  struct NvInferDeleter
  {
    template <typename T>
    void operator()(T* obj) const { if (obj) obj->destroy(); }
  };

  constexpr long double operator"" _GB(long double val) { return val * (1 << 30); }
  constexpr long double operator"" _MB(long double val) { return val * (1 << 20); }
  constexpr long double operator"" _KB(long double val) { return val * (1 << 10); }

  // These is necessary if we want to be able to write 1_GB instead of 1.0_GB.
  // Since the return type is signed, -1_GB will work as expected.
  constexpr long long int operator"" _GB(long long unsigned int val) { return val * (1 << 30); }
  constexpr long long int operator"" _MB(long long unsigned int val) { return val * (1 << 20); }
  constexpr long long int operator"" _KB(long long unsigned int val) { return val * (1 << 10); }

  void getDeviceMemoryInfo(size_t& free, size_t& total)
  {
    checkCudaError(FalcorCUDA::cudaMemGetInfo(&free, &total));
    free = static_cast<size_t> (free / (1024.f * 1024.f));
    total = static_cast<size_t> (total / (1024.f * 1024.f));
  }

  struct CuEventTimer
  {
    private:
        FalcorCUDA::cudaEvent_t eventStart;
        FalcorCUDA::cudaEvent_t eventEnd;
        FalcorCUDA::cudaStream_t stream;

    public:
      CuEventTimer(FalcorCUDA::cudaStream_t stream) : stream(stream)
      {
        checkCudaError(FalcorCUDA::cudaEventCreate(&eventStart));
        checkCudaError(FalcorCUDA::cudaEventCreate(&eventEnd));
      }

      ~CuEventTimer()
      {
        checkCudaError(FalcorCUDA::cudaEventDestroy(eventStart));
        checkCudaError(FalcorCUDA::cudaEventDestroy(eventEnd));
      }

      void start()
      {
        checkCudaError(FalcorCUDA::cudaEventRecord(eventStart, stream));
      }

      void end()
      {
        checkCudaError(FalcorCUDA::cudaEventRecord(eventEnd, stream));
      }

      void getElapsedTime(float& time)
      {
        checkCudaError(FalcorCUDA::cudaEventElapsedTime(&time, eventStart, eventEnd));
      }

      void sync()
      {
        checkCudaError(FalcorCUDA::cudaEventSynchronize(eventEnd));
      }
  };

  auto getNvInferEngineBuilders()
  {
    auto pBuilder = std::unique_ptr<nvinfer1::IBuilder, NvInferDeleter>(nvinfer1::createInferBuilder(gLogger));
    if (!pBuilder) throw std::runtime_error("Error creating nvinfer1::IBuilder");

    auto pNetwork = std::unique_ptr<nvinfer1::INetworkDefinition, NvInferDeleter>(pBuilder->createNetwork());
    if (!pNetwork) throw std::runtime_error("Error creating nvinfer1::INetworkDefinition");

    auto pParser = std::unique_ptr<nvuffparser::IUffParser, NvInferDeleter>(nvuffparser::createUffParser());
    if (!pParser) throw std::runtime_error("Error creating nvinfer1::IUffParser");

    return std::make_tuple(std::move(pBuilder), std::move(pNetwork), std::move(pParser));
  }

  void sanityCheck(bool condition, const char* message)
  {
    if (!condition) throw std::runtime_error("Sanity check failed: " + std::string(message));
  }

  nvinfer1::IShuffleLayer* createTransposeOutputLayer(nvinfer1::ITensor* tensor, ivec3 size, nvinfer1::INetworkDefinition* network)
  {
    auto layerTranspose = network->addShuffle(*tensor);
    layerTranspose->setReshapeDimensions(nvinfer1::DimsCHW(size.y, size.x, size.z));
    layerTranspose->setFirstTranspose(nvinfer1::Permutation{ 1, 2, 0 });
    network->markOutput(*layerTranspose->getOutput(0));
    return layerTranspose;
  }

  nvinfer1::IShuffleLayer* createTransposeInputLayer(nvinfer1::ITensor* tensor, ivec3 size, nvinfer1::INetworkDefinition* network)
  {
    auto layerTranspose = network->addShuffle(*tensor);
    layerTranspose->setReshapeDimensions(nvinfer1::DimsNCHW(1, size.z, size.y, size.x));
    layerTranspose->setFirstTranspose(nvinfer1::Permutation{ 0, 3, 1, 2 });
    network->markOutput(*layerTranspose->getOutput(0));
    return layerTranspose;
  }

  nvinfer1::ICudaEngine* createInputEngine(ivec2 windowSize, bool fp16)
  {
      std::unique_ptr<nvinfer1::IBuilder, NvInferDeleter> builder;
      std::unique_ptr<nvinfer1::INetworkDefinition, NvInferDeleter> network;
      std::unique_ptr<nvuffparser::IUffParser, NvInferDeleter> parser;
      std::tie(builder, network, parser) = getNvInferEngineBuilders();

      // Builder settings
      builder->setMaxBatchSize(1);
      builder->setMaxWorkspaceSize(100_MB);
      builder->setFp16Mode(fp16);

      // Define inputs
      const ivec2 sizeX = windowSize;
      const ivec2 sizeAux = windowSize;
      nvinfer1::ITensor* tensorColor = network->addInput("in_image", nvinfer1::DataType::kFLOAT, nvinfer1::DimsNCHW{ 1, sizeX.y , sizeX.x, 3 });
      nvinfer1::ITensor* tensorAux = network->addInput("in_aux", nvinfer1::DataType::kFLOAT, nvinfer1::DimsNCHW{ 1, sizeAux.y, sizeAux.x, 4 });

      // Create transposed outputs
      createTransposeInputLayer(tensorColor, ivec3(sizeX, 3), network.get());
      createTransposeInputLayer(tensorAux, ivec3(sizeAux, 4), network.get());

      // Sanity checks
      sanityCheck(network->getNbInputs() == 2, "network input bindings mismatch");
      sanityCheck(network->getNbOutputs() == 2, "network output bindings mismatch");

      // Create engine
      return builder->buildCudaEngine(*network);
  }
}

TrtRdae::TrtRdae()
{
  checkCudaError(FalcorCUDA::cudaStreamCreate(&mStream));
}

TrtRdae::~TrtRdae()
{
  checkCudaError(FalcorCUDA::cudaStreamDestroy(mStream));
}

void TrtRdae::create(ivec2 size, bool fp16)
{
  //--------------------------------------------------------------------------
  // Load/Create inference engines
  //--------------------------------------------------------------------------
  mpInputEngine.reset();
  mpFilterEngine.reset();

  // Create input engine
  mpInputEngine = InferenceEngine::create(createInputEngine(size, fp16), "input");
  sanityCheck(mpInputEngine.get(), "no engine created");

  // Create filter engine
  mpFilterEngine = InferenceEngine::create("Data/filter_fp16_1280x768", "rdae");
  sanityCheck(mpFilterEngine.get(), "no engine created");

  // Allocate device memory
  mTotalBufferDeviceMemory = setupCudaBuffers();

  // Setup raw pointer buffers for inference
  setupRawBuffers();

  // Sum up needed device memory for all inference engines
  mTotalEngineDeviceMemory = 0;
  mTotalEngineDeviceMemory += mpInputEngine  ? mpInputEngine->getEngineDeviceMemoryInMB()  : 0;
  mTotalEngineDeviceMemory += mpFilterEngine ? mpFilterEngine->getEngineDeviceMemoryInMB() : 0;

  Falcor::logInfo("TensorRT Engine created...");
}

size_t TrtRdae::setupCudaBuffers()
{
  // Allocate cuda buffers for inference engines
  std::vector<CudaBuffer<void>> generated_buffers;

  // Generate cuda buffers for input network
  {
    generated_buffers = mpInputEngine->generateCudaBuffers(false);
    sanityCheck(generated_buffers.size() == 2, "amount of generated does not match");
    mBufferInputImage = std::move(generated_buffers.at(0));
    mBufferInputAux = std::move(generated_buffers.at(1));
  }

  // Generate cuda buffers for filter network
  {
    generated_buffers = mpFilterEngine->generateCudaBuffers(false);
    mBuffersRecurrent.clear();
    for (int i = 0; i < generated_buffers.size() - 1; i++)
      mBuffersRecurrent.push_back(std::move(generated_buffers.at(i)));
  }

  // Get total amount of allocated cuda device buffer memory
  size_t totalAllocatedBufferSizeinMB = 0;
  totalAllocatedBufferSizeinMB += mpInputEngine ? mpInputEngine->getAllocatedBufferMemoryMB() : 0;
  totalAllocatedBufferSizeinMB += mpFilterEngine ? mpFilterEngine->getAllocatedBufferMemoryMB() : 0;
  return totalAllocatedBufferSizeinMB;
}

void TrtRdae::setupRawBuffers()
{
  // Input engine
  mRawInputBuffers.resize(4);
  mRawInputBuffers[0] = nullptr;            //  In: (HWC) Input Color [set during inference with cuda buffers]
  mRawInputBuffers[1] = nullptr;            //  In: (HWC) Auxilliary  [set during inference with cuda buffers]
  mRawInputBuffers[2] = mBufferInputImage;  // Out: (CHW) Input Color
  mRawInputBuffers[3] = mBufferInputAux;    // Out: (CHW) Auxilliary

  // Create raw cuda pointer buffer vectors for inference
  mRawFilterBuffers.clear();
  mRawFilterBuffers.push_back(mBufferInputImage.data()); //  In: (CHW) Filter Input Color
  mRawFilterBuffers.push_back(mBufferInputAux.data());   //  In: (CHW) Filter Auxilliary input
  for (int i = 0; i < 2; i++)
    for (const auto& buffer : mBuffersRecurrent)
      mRawFilterBuffers.push_back(buffer.data());        // In/Out: (CHW) Recurrent connections
  mRawFilterBuffers.push_back(nullptr);                  // Out: (CHW) Filter Output Color
}

bool TrtRdae::infer(CudaExternalMemory& inColor, CudaExternalMemory& inAux, CudaExternalMemory& outColor, bool clearRecurrentBuffers)
{
  if (!mpInputEngine || !mpFilterEngine)
    return false;

  // Assign cuda buffers to in/outs
  mRawInputBuffers[0] = inColor.data();
  mRawInputBuffers[1] = inAux.data();
  //... inbetween are the recurrent buffers
  mRawFilterBuffers.back() = outColor.data();

  if (clearRecurrentBuffers)
  {
    for (auto& buf : mBuffersRecurrent)
      buf.memset(0);
  }

  // Execute engines
  mpInputEngine->executeAsync(mRawInputBuffers.data(), mStream);
  mpFilterEngine->executeAsync(mRawFilterBuffers.data(), mStream);

  getDeviceMemoryInfo(mFreeDeviceMemory, mTotalDeviceMemory);
  return true;
}
