#include "Rdae.h"
#include "FalcorCUDA.h"

const char* Rdae::kDesc = "Rdae";

namespace
{
    const ivec2 kCNNSize = ivec2(1280, 768);

    const char kPrepareInputShaderFile[]  = "Passes/Rdae/PrepareRdaeInput.cs.slang";
    const char kPrepareOutputShaderFile[] = "Passes/Rdae/PrepareRdaeOutput.cs.slang";
}

Rdae::SharedPtr Rdae::create()
{
  Rdae::SharedPtr pPass(new Rdae());
  return pPass;
}

Rdae::Rdae()
{
    mpCudaFence = CudaDx12Fence::create();
    mpComputeState = ComputeState::create();
    mRdaeTimer.setStream(mTrtRdae.getCudaStream());

    // Create inference engine
    mTrtRdae.create(kCNNSize, true);
}

void Rdae::onLoad(RenderContext* pRenderContext, PassData& passData)
{
    createPrograms();
    createResources(passData);
}

void Rdae::onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData)
{
    createPrograms();
    createResources(passData);
}

void Rdae::createPrograms()
{
    mpPrepareInputProgram = ComputeProgram::createFromFile(kPrepareInputShaderFile, "main");
    mpPrepareInputVars = ComputeVars::create(mpPrepareInputProgram->getReflector());

    mpPrepareOutputProgram = ComputeProgram::createFromFile(kPrepareOutputShaderFile, "main");
    mpPrepareOutputVars = ComputeVars::create(mpPrepareOutputProgram->getReflector());
}

void Rdae::createResources(PassData& passData)
{
    const int width = passData.getWidth();
    const int height = passData.getHeight();

    auto bufferBindFlags = Buffer::BindFlags::UnorderedAccess | Buffer::BindFlags::ShaderResource | Buffer::BindFlags::Shared;
    mpBufferRdaeInput  = StructuredBuffer::create(mpPrepareInputProgram,  "gRdaeInput",  kCNNSize.x * kCNNSize.y, bufferBindFlags);
    mpBufferRdaeAux    = StructuredBuffer::create(mpPrepareInputProgram,  "gRdaeAux",    kCNNSize.x * kCNNSize.y, bufferBindFlags);
    mpBufferRdaeOutput = StructuredBuffer::create(mpPrepareOutputProgram, "gRdaeOutput", kCNNSize.x * kCNNSize.y, bufferBindFlags);

    auto pOutput = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget);
    passData.addResource("gRdaeOutput", pOutput);
}

void Rdae::onDataReload()
{
    createPrograms();
}

void Rdae::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
}

void Rdae::onGuiRender(Gui* pGui)
{
    //pGui->addCheckBox("Clear recurrent buffers", mClearRecurrentBuffers);
    //pGui->addCheckBox("Clear auxiliary buffers", mClearAux);

    static bool showCudaTimings = false;
    pGui->addCheckBox("Show inference time", showCudaTimings);
    pGui->addTooltip("Shows CUDA inference time in ms", true);
    if (showCudaTimings)
    {
        mRdaeTimer.sync();
        pGui->addText((std::string("Inference time -> ") + std::to_string(mRdaeTimer.getElapsedTime()) + "ms").c_str());
    }

    if (pGui->beginGroup("GPU memory", false))
    {
        pGui->addText("Total: ");
        pGui->addText((std::to_string(mTrtRdae.mTotalDeviceMemory) + "MB").c_str(), true);
        pGui->addText("Free: ");
        pGui->addText((std::to_string(mTrtRdae.mFreeDeviceMemory) + "MB").c_str(), true);
        pGui->addSeparator();
        pGui->addText("Engines: ");
        pGui->addText((std::to_string(mTrtRdae.mTotalEngineDeviceMemory) + "MB").c_str(), true);
        pGui->addSeparator();
        pGui->addText("Buffers: ");
        pGui->addText((std::to_string(mTrtRdae.mTotalBufferDeviceMemory) + "MB").c_str(), true);
        pGui->endGroup();
    }
}

void Rdae::onFrameRender(RenderContext* pRenderContext, PassData& passData)
{
    PROFILE("Rdae");

    Texture::SharedPtr pAlbedo = asTexture(passData["gAlbedo"]);
    Texture::SharedPtr pColor  = asTexture(passData["gCombined"]);
    Texture::SharedPtr pAux    = asTexture(passData["gCNNAux"]);
    Texture::SharedPtr pOut    = asTexture(passData["gRdaeOutput"]);

    const ivec2 windowSize = ivec2(pColor->getWidth(), pColor->getHeight());

    if (windowSize.x != 1280 || windowSize.y != 720) // Only 720p!
        return;

    if (!mExtCudaBufferColor.isMapped(mpBufferRdaeInput))   mExtCudaBufferColor  = CudaExternalMemory::create(mpBufferRdaeInput);
    if (!mExtCudaBufferAux.isMapped(mpBufferRdaeAux))       mExtCudaBufferAux    = CudaExternalMemory::create(mpBufferRdaeAux);
    if (!mExtCudaBufferOutput.isMapped(mpBufferRdaeOutput)) mExtCudaBufferOutput = CudaExternalMemory::create(mpBufferRdaeOutput);

    {
        PROFILE("PrepareInput");

        mpPrepareInputVars->setTexture("gAlbedo",   pAlbedo);
        mpPrepareInputVars->setTexture("gInColor",  pColor);
        mpPrepareInputVars->setTexture("gInCNNAux", pAux);

        mpPrepareInputVars->setStructuredBuffer("gRdaeInput", mpBufferRdaeInput);
        mpPrepareInputVars->setStructuredBuffer("gRdaeAux", mpBufferRdaeAux);

        mpPrepareInputVars["PerFrameCB"]["gExponent"]   = mExponent;
        mpPrepareInputVars["PerFrameCB"]["gWindowDims"] = windowSize;
        mpPrepareInputVars["PerFrameCB"]["gCNNDims"]    = kCNNSize;

        const glm::uvec3 numGroups = div_round_up(glm::uvec3(kCNNSize.x, kCNNSize.y, 1u), mpPrepareInputProgram->getReflector()->getThreadGroupSize());

        mpComputeState->setProgram(mpPrepareInputProgram);
        pRenderContext->setComputeState(mpComputeState);
        pRenderContext->setComputeVars(mpPrepareInputVars);
        pRenderContext->dispatch(numGroups.x, numGroups.y, numGroups.z);
    }

    {
        PROFILE("Inference");

        auto cudaStream = mTrtRdae.getCudaStream();
        auto commandQueue = pRenderContext->getLowLevelData()->getCommandQueue();

        pRenderContext->flush(false);

        mpCudaFence->signalCommandQueue(commandQueue);
        mpCudaFence->waitStream(cudaStream);

        if (mClearAux) mExtCudaBufferAux.memset(0);

        mRdaeTimer.start();
        mTrtRdae.infer(mExtCudaBufferColor, mExtCudaBufferAux, mExtCudaBufferOutput, mClearRecurrentBuffers);
        mRdaeTimer.end();

        mpCudaFence->signalStream(cudaStream);
        mpCudaFence->waitCommandQueue(commandQueue);
    }

    {
        PROFILE("PrepareOutput");

        mpPrepareOutputVars->setStructuredBuffer("gRdaeOutput", mpBufferRdaeOutput);
        mpPrepareOutputVars->setTexture("gOutput", pOut);

        mpPrepareOutputVars["PerFrameCB"]["gExponent"]   = 1.f / mExponent;
        mpPrepareOutputVars["PerFrameCB"]["gWindowDims"] = windowSize;
        mpPrepareOutputVars["PerFrameCB"]["gCNNDims"]    = kCNNSize;

        const glm::uvec3 numGroups = div_round_up(glm::uvec3(kCNNSize.x, kCNNSize.y, 1u), mpPrepareOutputProgram->getReflector()->getThreadGroupSize());

        mpComputeState->setProgram(mpPrepareOutputProgram);
        pRenderContext->setComputeState(mpComputeState);
        pRenderContext->setComputeVars(mpPrepareOutputVars);
        pRenderContext->dispatch(numGroups.x, numGroups.y, numGroups.z);
    }
}
