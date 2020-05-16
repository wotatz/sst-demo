#include "GBuffer.h"
#include "GBufferData.h"

const char* GBuffer::kDesc = "GBuffer";

namespace
{
  const char kFileGBufferRasterized[] = "Passes/GBuffer/GBuffer.hlsl";
};

GBuffer::SharedPtr GBuffer::create()
{
  GBuffer::SharedPtr pPass(new GBuffer());
  return pPass;
}

GBuffer::GBuffer()
{
  mpFbo          = Fbo::create();
  mRaster.pState = GraphicsState::create();

  createPrograms();
}

void GBuffer::onDataReload()
{
  createPrograms();
}

void GBuffer::createPrograms()
{
  mRaster.pProgram = GraphicsProgram::createFromFile(kFileGBufferRasterized, "vs", "ps");
  mRaster.pVars    = GraphicsVars::create(mRaster.pProgram->getReflector());
  mRaster.pState->setProgram(mRaster.pProgram);

  RasterizerState::Desc rsDesc;
  rsDesc.setCullMode(mRaster.cullMode);
  mRaster.pState->setRasterizerState(RasterizerState::create(rsDesc));
}

void GBuffer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
  mpScene = pScene;
  mpSceneRenderer = (pScene == nullptr) ? nullptr : SceneRenderer::create(pScene);
}

void GBuffer::onLoad(RenderContext* pRenderContext, PassData& passData)
{
    createResources(passData);
}

void GBuffer::onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData)
{
    createResources(passData);
}

void GBuffer::createResources(PassData& passData)
{
    // Create resources
    for (int i = 0; i < kGBufferChannelDesc.size(); ++i)
    {
        auto pTexture = Texture::create2D(passData.getWidth(), passData.getHeight(), kGBufferChannelDesc[i].format, 1u, 1u, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget);
        passData.addResource(kGBufferChannelDesc.at(i).texname, pTexture);
    }
    auto pDepth = Texture::create2D(passData.getWidth(), passData.getHeight(), ResourceFormat::D32Float, 1u, 1u, nullptr, Resource::BindFlags::DepthStencil | Resource::BindFlags::ShaderResource);
    passData.addResource("gDepth", pDepth);

    // Attach textures to framebuffer
    for (int i = 0; i < kGBufferChannelDesc.size(); ++i)
        mpFbo->attachColorTarget(asTexture(passData[kGBufferChannelDesc.at(i).texname]), i);

    mpFbo->attachDepthStencilTarget(asTexture(passData["gDepth"]));
}

void GBuffer::setPatternGenerator(const PatternGenerator::SharedPtr& pGenerator)
{
    mpSceneRenderer->getScene()->getActiveCamera()->setPatternGenerator(pGenerator, 1.0f / vec2(mpFbo->getWidth(), mpFbo->getHeight()));
}

void GBuffer::onFrameRender(RenderContext* pRenderContext, PassData& passData)
{
  PROFILE("GBuffer");

  if (!mpScene) return;

  ConstantBuffer::SharedPtr pCB = mRaster.pVars->getConstantBuffer("PerFrameCB");
  pCB["gRenderTargetDim"] = passData.getExtend();

  // Clear render targets
  pRenderContext->clearFbo(mpFbo.get(), vec4(0), 1.f, 0, FboAttachmentType::All);
  mRaster.pState->setFbo(mpFbo);

  pRenderContext->setGraphicsState(mRaster.pState);
  pRenderContext->setGraphicsVars(mRaster.pVars);

  Camera::SharedPtr pCamera = mpScene->getActiveCamera();
  if (pCamera)
      mpSceneRenderer->renderScene(pRenderContext, pCamera.get());
}

void GBuffer::onGuiRender(Gui* pGui)
{
}
