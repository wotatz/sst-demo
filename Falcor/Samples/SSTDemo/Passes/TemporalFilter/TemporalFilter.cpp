#include "TemporalFilter.h"
#include "../gbuffer/GBufferData.h"

const char* TemporalFilter::kDesc = "Temporal filtering";

namespace
{
  // Shaders
  const char *kGradientEstimationShader   = "Passes/TemporalFilter/GradientEstimation.slang";
  const char *kTemporalAccumulationShader = "Passes/TemporalFilter/TemporalAccumulation.slang";
}

TemporalFilter::SharedPtr TemporalFilter::create()
{
    TemporalFilter::SharedPtr pPass(new TemporalFilter());
  return pPass;
}

TemporalFilter::TemporalFilter()
{
  mpState = GraphicsState::create();
  createPrograms();
}

void TemporalFilter::onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData)
{
    createResources(passData);
}

void TemporalFilter::onLoad(RenderContext* pRenderContext, PassData& passData)
{
    createResources(passData);
}
void TemporalFilter::onDataReload()
{
    createPrograms();
}

void TemporalFilter::clearFbos(RenderContext* pContext)
{
  pContext->clearFbo(mpCurTemporalAccFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);
  pContext->clearFbo(mpPrevTemporalAccFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);
  mClearFBOs = false;
}

void TemporalFilter::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
}

void TemporalFilter::createPrograms()
{
  mpPassGradEst     = FullScreenPass::create(kGradientEstimationShader);
  mpCurGradEstVars  = GraphicsVars::create(mpPassGradEst->getProgram()->getReflector());
  mpPrevGradEstVars = GraphicsVars::create(mpPassGradEst->getProgram()->getReflector());

  mpPassTempAcc = FullScreenPass::create(kTemporalAccumulationShader);
  mpCurVars     = GraphicsVars::create(mpPassTempAcc->getProgram()->getReflector());
  mpPrevVars    = GraphicsVars::create(mpPassTempAcc->getProgram()->getReflector());
}

void TemporalFilter::createResources(PassData& passData)
{
    const int width  = passData.getWidth();
    const int height = passData.getHeight();

    // Create textures
    mpPrevLinearZ = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);

    Texture::SharedPtr pFiltered = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);
    passData.addResource("gTemporalFiltered", pFiltered);

    // Create Gradient Estimation FBO
    {
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);  // prevDirect
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA16Float);  // history length, Normalized Gradient, validity
        mpCurGradEstFbo = FboHelper::create2D(width, height, desc);
        mpPrevGradEstFbo = FboHelper::create2D(width, height, desc);
    }

    // Create Temporal Accumulation FBOs
    {
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);  // output
        desc.setColorTarget(1, Falcor::ResourceFormat::RGBA32Float);  // modulated output
        mpCurTemporalAccFbo = FboHelper::create2D(width, height, desc);
        mpPrevTemporalAccFbo = FboHelper::create2D(width, height, desc);
    }

    mClearFBOs = true;
}

void TemporalFilter::onGuiRender(Gui* pGui)
{
    pGui->addFloatVar("BaseExponent", mBaseExponent, 1.f, 10.f, 0.1f);
    pGui->addTooltip("Controls the blending with previous frames", true);
    pGui->addCheckBox("Modulate", mModulate);
    pGui->addTooltip("Enable/Disable albedo modulation", true);
}

void TemporalFilter::onFrameRender(RenderContext* pRenderContext, PassData& passData)
{
  PROFILE("TemporalFilter");

  if (mClearFBOs)
    clearFbos(pRenderContext);

  computeGradientEstiamtion(pRenderContext, passData);
  computeTemporalAccumulation(pRenderContext, passData);

  // Get textures
  Texture::SharedPtr pLinearZ = asTexture(passData["gLinearZAndNormal"]);
  Texture::SharedPtr pOutput  = asTexture(passData["gTemporalFiltered"]);

  // Blit to output
  pRenderContext->blit(mpCurTemporalAccFbo->getColorTexture(1)->getSRV(), pOutput->getRTV(), uvec4(-1), uvec4(-1), Sampler::Filter::Point);

  // Swap FBOs for next pass
  std::swap(mpCurGradEstFbo, mpPrevGradEstFbo);
  std::swap(mpCurGradEstVars, mpPrevGradEstVars);

  std::swap(mpCurTemporalAccFbo, mpPrevTemporalAccFbo);
  std::swap(mpCurVars, mpPrevVars);

  // Store previous linearZ
  pRenderContext->blit(pLinearZ->getSRV(), mpPrevLinearZ->getRTV(), uvec4(-1), uvec4(-1), Sampler::Filter::Point);
}

void TemporalFilter::computeGradientEstiamtion(RenderContext* pRenderContext, PassData& passData)
{
  // Get textures
  Texture::SharedPtr pIllumination     = asTexture(passData["gRdaeOutput"]);
  Texture::SharedPtr pMotion           = asTexture(passData["gMotion"]);
  Texture::SharedPtr pLinearZAndNormal = asTexture(passData["gLinearZAndNormal"]);
  Texture::SharedPtr pPosNormalFwidth  = asTexture(passData["gPosNormalFwidth"]);

  mpCurGradEstVars->setTexture("gIllumination", pIllumination);
  mpCurGradEstVars->setTexture("gPrevIllumination", mpPrevTemporalAccFbo->getColorTexture(0));
  mpCurGradEstVars->setTexture("gAccHistory", mpPrevGradEstFbo->getColorTexture(1));
  mpCurGradEstVars->setTexture("gMotion", pMotion);
  mpCurGradEstVars->setTexture("gPosNormalFwidth", pPosNormalFwidth);
  mpCurGradEstVars->setTexture("gLinearZAndNormal", pLinearZAndNormal);
  mpCurGradEstVars->setTexture("gPrevLinearZAndNormal", mpPrevLinearZ);

  // Execute pass
  mpState->setFbo(mpCurGradEstFbo);
  pRenderContext->pushGraphicsState(mpState);
  pRenderContext->pushGraphicsVars(mpCurGradEstVars);
  mpPassGradEst->execute(pRenderContext);
  pRenderContext->popGraphicsVars();
  pRenderContext->popGraphicsState();
}

void TemporalFilter::computeTemporalAccumulation(RenderContext* pRenderContext, PassData& passData)
{
  // Get textures
  Texture::SharedPtr pIllumination = asTexture(passData["gRdaeOutput"]);
  Texture::SharedPtr pMotion = asTexture(passData["gMotion"]);
  Texture::SharedPtr pAlbedo = asTexture(passData["gAlbedo"]);

  // Set constant buffer
  ConstantBuffer::SharedPtr pCB = mpCurVars->getConstantBuffer("PerFrameCB");
  pCB["gBaseExponent"] = mBaseExponent;
  pCB["gModulate"]     = mModulate;

  // Setup textures
  mpCurVars->setTexture("gIllumination", pIllumination);
  mpCurVars->setTexture("gPrevIllumination", mpCurGradEstFbo->getColorTexture(0));
  mpCurVars->setTexture("gAccHistory", mpCurGradEstFbo->getColorTexture(1));
  mpCurVars->setTexture("gAlbedo", pAlbedo);

  // Execute pass
  mpState->setFbo(mpCurTemporalAccFbo);
  pRenderContext->pushGraphicsState(mpState);
  pRenderContext->pushGraphicsVars(mpCurVars);
  mpPassTempAcc->execute(pRenderContext);
  pRenderContext->popGraphicsVars();
  pRenderContext->popGraphicsState();
}
