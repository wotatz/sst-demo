#pragma once
#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/BasePass.h"
#include "Passes/Shared/VPLData.h"

using namespace Falcor;

class VPLSampling : public BasePass
{
public:
    using SharedPtr = std::shared_ptr<VPLSampling>;

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
    VPLSampling();
    void createPrograms();
    void createResources(PassData& passData);
    void toogleProgramDefine(bool enabled, std::string name, std::string value = "");

    // Ray tracing program.
    struct
    {
        RtProgram::SharedPtr       pProgram;
        RtProgramVars::SharedPtr   pVars;
        RtSceneRenderer::SharedPtr pSceneRenderer;
    } mTracer;

    RtState::SharedPtr mpState;
    RtScene::SharedPtr mpScene;

    // Various internal parameters
    uint32_t mFrameCount  = 0x1337u;  // Frame counter to vary random numbers over time
    bool mReloadResources = false;
    mat4 mLastCameraMatrix;

    // BRDF settings
    bool mUseDirectGGX = true;
    bool mUseIndirectGGX = false;

    // Sampling settings
    bool  mEnableDirectSampling = true;
    bool  mEnableVPLSampling    = true;
    bool  mAccumulateSamples    = false;
    bool  mUseUniformSampling   = false;

    int   mNumDirectSamples      = 1;
    int   mNumVPLSamples         = 1;
    int   mNumAccumulatedSamples = 0;

    float mMinT               = 0.01f;
    float mGMax               = 10.f;
    float mAttenuationEpsilon = 0.05f;
};
