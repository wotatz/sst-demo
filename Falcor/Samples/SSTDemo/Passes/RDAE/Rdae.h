#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/BasePass.h"
#include "Passes/Shared/VPLData.h"

#include "TrtRdae.h"
#include "Utils/Cuda/CudaDx12Fence.h"
#include "Utils/Cuda/CudaExternalMemory.h"
#include "Utils/Cuda/CudaTimer.h"

using namespace Falcor;


class Rdae : public BasePass
{
public:
    using SharedPtr = std::shared_ptr<Rdae>;

    static SharedPtr create();

    virtual void onFrameRender(RenderContext* pRenderContext, PassData& passData) override;
    virtual void onLoad(RenderContext* pRenderContext, PassData& passData) override;
    virtual void onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual void onDataReload() override;
    virtual void onGuiRender(Gui* pGui) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;

    static const char* kDesc;
    virtual std::string getDesc() override { return kDesc; }

private:
    Rdae();
    void createPrograms();
    void createResources(PassData& passData);

    // Gui variables
    bool mClearRecurrentBuffers = false;
    bool mClearAux              = false;

    float mExponent = 0.2f;

    ComputeProgram::SharedPtr mpPrepareInputProgram;
    ComputeVars::SharedPtr    mpPrepareInputVars;

    ComputeProgram::SharedPtr mpPrepareOutputProgram;
    ComputeVars::SharedPtr    mpPrepareOutputVars;

    ComputeState::SharedPtr   mpComputeState;

    StructuredBuffer::SharedPtr mpBufferRdaeInput;
    StructuredBuffer::SharedPtr mpBufferRdaeAux;
    StructuredBuffer::SharedPtr mpBufferRdaeOutput;

    CudaExternalMemory mExtCudaBufferColor;
    CudaExternalMemory mExtCudaBufferAux;
    CudaExternalMemory mExtCudaBufferOutput;

    CudaDx12Fence::SharedPtr mpCudaFence;
    TrtRdae mTrtRdae;
    CuEventTimer mRdaeTimer;
};
