#pragma once
#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/BasePass.h"


using namespace Falcor;

class GBuffer : public BasePass
{
public:
    using SharedPtr = std::shared_ptr<GBuffer>;

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

    void setPatternGenerator(const PatternGenerator::SharedPtr& pGenerator);

private:
    GBuffer();
    void createPrograms();
    void createResources(PassData& passData);

    Scene::SharedPtr            mpScene;
    SceneRenderer::SharedPtr    mpSceneRenderer;
    Fbo::SharedPtr              mpFbo;

    struct
    {
        GraphicsState::SharedPtr   pState;
        GraphicsProgram::SharedPtr pProgram;
        GraphicsVars::SharedPtr    pVars;
        RasterizerState::CullMode  cullMode = RasterizerState::CullMode::Back;
    } mRaster;
};
