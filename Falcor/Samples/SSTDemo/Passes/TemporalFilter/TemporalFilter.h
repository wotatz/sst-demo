#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/BasePass.h"
#include "Passes/Shared/VPLData.h"

using namespace Falcor;

class TemporalFilter : public BasePass
{
public:
    using SharedPtr = std::shared_ptr<TemporalFilter>;

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
    TemporalFilter();
    void createPrograms();
    void createResources(PassData& passData);
    void clearFbos(RenderContext* pContext);

    void computeGradientEstiamtion(RenderContext* pRenderContext, PassData& passData);
    void computeTemporalAccumulation(RenderContext* pRenderContext, PassData& passData);

    // Some common pass bookkeeping
    bool mClearFBOs     = false;

    // Gui variables
    float mBaseExponent = 2.f;
    bool  mModulate     = true;

    // Forward Pass state and variables
    GraphicsState::SharedPtr    mpState;
    FullScreenPass::UniquePtr   mpPassGradEst;
    FullScreenPass::UniquePtr   mpPassTempAcc;

    Fbo::SharedPtr              mpCurGradEstFbo;
    Fbo::SharedPtr              mpPrevGradEstFbo;
    GraphicsVars::SharedPtr     mpCurGradEstVars;
    GraphicsVars::SharedPtr     mpPrevGradEstVars;

    Fbo::SharedPtr              mpCurTemporalAccFbo;
    Fbo::SharedPtr              mpPrevTemporalAccFbo;
    GraphicsVars::SharedPtr     mpCurVars;
    GraphicsVars::SharedPtr     mpPrevVars;

    Texture::SharedPtr mpPrevLinearZ;
};
