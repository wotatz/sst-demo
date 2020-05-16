#include "VPLSampling.h"
#include "../Shared/VPLTreeStructs.h"

const char* VPLSampling::kDesc = "VPL Sampling";

namespace
{
  // Lightcuts rt-shader file path
  const char *kFileRayTrace = "Passes/VPLSampling/VPLSampling.rt.hlsl";

  // Lightcuts rt-shader entry points
  const char* kEntryPointRayGen   = "rayGeneration";
  const char* kEntryPointMiss0    = "shadowRayMiss";
  const char* kEntryPointAnyHit0  = "shadowRayAnyHit";
}

VPLSampling::SharedPtr VPLSampling::create()
{
    VPLSampling::SharedPtr pPass(new VPLSampling());
    return pPass;
}

VPLSampling::VPLSampling()
{
    createPrograms();
}

void VPLSampling::createPrograms()
{
  if (!mpScene) return;

  // Create ray tracing program
  RtProgram::Desc desc;
  desc.addShaderLibrary(kFileRayTrace);
  desc.setRayGen(kEntryPointRayGen);

  // Add ray type #0 (shadow rays)
  desc.addMiss(0, kEntryPointMiss0);
  desc.addHitGroup(0, "", kEntryPointAnyHit0);

  mTracer.pProgram = RtProgram::create(desc);
  mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mpScene);

  mpState = RtState::create();
  mpState->setMaxTraceRecursionDepth(1);
  mpState->setProgram(mTracer.pProgram);
}

void VPLSampling::createResources(PassData& passData)
{
    // Assign dummy resources so that the textures will show up in the dropdown menu...
    Resource::SharedPtr pDummy;
    passData.addResource("gAlbedo", pDummy);
    passData.addResource("gCombined", pDummy);
    passData.addResource("gDirect", pDummy);
    passData.addResource("gIndirect", pDummy);

    if (!mpScene) return;
    if (!mTracer.pProgram) return;

    createPrograms();

    auto bindFlags  = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget;
    auto pAlbedo    = Texture::create2D(passData.getWidth(), passData.getHeight(), ResourceFormat::RGBA32Float, 1u, 1u, nullptr, bindFlags);
    auto pCombined  = Texture::create2D(passData.getWidth(), passData.getHeight(), ResourceFormat::RGBA32Float, 1u, 1u, nullptr, bindFlags);
    auto pDirect    = Texture::create2D(passData.getWidth(), passData.getHeight(), ResourceFormat::RGBA32Float, 1u, 1u, nullptr, bindFlags);
    auto pIndirect  = Texture::create2D(passData.getWidth(), passData.getHeight(), ResourceFormat::RGBA32Float, 1u, 1u, nullptr, bindFlags);

    passData.addResource("gAlbedo", pAlbedo);
    passData.addResource("gCombined", pCombined);
    passData.addResource("gDirect", pDirect);
    passData.addResource("gIndirect", pIndirect);

    mReloadResources = false;
}

void VPLSampling::toogleProgramDefine(bool enabled, std::string name, std::string value)
{
    if (enabled)
        mTracer.pProgram->addDefine(name, value);
    else
        mTracer.pProgram->removeDefine(name);
}

void VPLSampling::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = std::dynamic_pointer_cast<RtScene>(pScene);
    if (mpScene)
    {
        mTracer.pSceneRenderer = RtSceneRenderer::create(mpScene);
        createPrograms();
        mReloadResources = true;
    }
}

void VPLSampling::onLoad(RenderContext* pRenderContext, PassData& passData)
{
    createPrograms();
    createResources(passData);
}

void VPLSampling::onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData)
{
    createResources(passData);
}

void VPLSampling::onDataReload()
{
    createPrograms();
}

void VPLSampling::onGuiRender(Gui* pGui)
{
    pGui->addCheckBox("Direct GGX", mUseDirectGGX);
    pGui->addTooltip("Enable/Disable GGX for direct lighting", true);
    pGui->addCheckBox("Indirect GGX", mUseIndirectGGX);
    pGui->addTooltip("Enable/Disable GGX for indirect lighting", true);
    pGui->addSeparator();
    pGui->addCheckBox("Enable direct sampling", mEnableDirectSampling);
    pGui->addCheckBox("Enable indirect sampling", mEnableVPLSampling);
    pGui->addSeparator();
    pGui->addCheckBox("Use uniform sampling", mUseUniformSampling);
    pGui->addTooltip("Use uniform sampling strategy by picking a random VPL", true);
   
    pGui->addFloatVar("minT", mMinT, 0.f, 10.f);
    pGui->addFloatVar("maxG", mGMax, 0.01f, 1000.f);
    pGui->addTooltip("Clamping term for singularties caused by squared falloff", true);
    pGui->addFloatVar("attenuation epsilon", mAttenuationEpsilon, 0.0001f, 1000.f);
    pGui->addTooltip("Hierarchical importance sampling inverse squared attenation lower bound", true);

    if (pGui->addCheckBox("Accumulate samples", mAccumulateSamples))
        if (mAccumulateSamples) mNumAccumulatedSamples = 0;
}

void VPLSampling::onFrameRender(RenderContext* pRenderContext, PassData& passData)
{
  PROFILE("VPLSampling");

  if (!mpScene)
      return;

  if (mReloadResources)
      createResources(passData);

  auto& pCamera = mpScene->getActiveCamera();

  if (pCamera->getViewMatrix() != mLastCameraMatrix)
      mNumAccumulatedSamples = 0;

  const int maxVPLs = passData.getVariable<int>("maxVPLs");
  const int numInternalNodes = getNumInternalNodes(maxVPLs);
  const int numTotalNodes    = getNumTotalNodes(maxVPLs);

  // Get resources
  Texture::SharedPtr pGBufferWorldPosition = asTexture(passData["gPosW"]);
  Texture::SharedPtr pGBufferPacked1       = asTexture(passData["gPacked1"]);
  Texture::SharedPtr pGBufferPacked2       = asTexture(passData["gPacked2"]);

  StructuredBuffer::SharedPtr pBufferVPLData  = asStructuredBuffer(passData["gVPLData"]);
  StructuredBuffer::SharedPtr pBufferVPLStats = asStructuredBuffer(passData["gVPLStats"]);

  Texture::SharedPtr pAlbedo             = asTexture(passData["gAlbedo"]);
  Texture::SharedPtr pCombined           = asTexture(passData["gCombined"]);
  Texture::SharedPtr pDirect             = asTexture(passData["gDirect"]);
  Texture::SharedPtr pIndirect           = asTexture(passData["gIndirect"]);

  // Bind resources
  auto globalVars = mTracer.pVars->getGlobalVars();
  globalVars->setTexture("gPosW", pGBufferWorldPosition);
  globalVars->setTexture("gPacked1", pGBufferPacked1);
  globalVars->setTexture("gPacked2", pGBufferPacked2);

  globalVars->setStructuredBuffer("gVPLData", pBufferVPLData);
  globalVars->setStructuredBuffer("gVPLStats", pBufferVPLStats);

  globalVars->setTexture("gAlbedo", pAlbedo);
  globalVars->setTexture("gCombined", pCombined);
  globalVars->setTexture("gDirect", pDirect);
  globalVars->setTexture("gIndirect", pIndirect);

  // Set constant buffer
  globalVars["CB"]["gGMax"]                  = mGMax;
  globalVars["CB"]["gAttenuationEpsilon"]    = mAttenuationEpsilon;
  globalVars["CB"]["gMaxVPLs"]               = maxVPLs;
  globalVars["CB"]["gNumDirectSamples"]      = mNumDirectSamples;
  globalVars["CB"]["gNumIndirectSamples"]    = mNumVPLSamples;
  globalVars["CB"]["gNumAccumulatedSamples"] = mNumAccumulatedSamples;

  globalVars["PerFrameCB"]["gMinT"]       = mMinT;
  globalVars["PerFrameCB"]["gFrameCount"] = mFrameCount;

  toogleProgramDefine(mUseDirectGGX,         "USE_DIRECT_GGX");
  toogleProgramDefine(mUseIndirectGGX,       "USE_INDIRECT_GGX");
  toogleProgramDefine(mEnableDirectSampling, "DIRECT_SAMPLING_ENABLED");
  toogleProgramDefine(mEnableVPLSampling,    "INDIRECT_SAMPLING_ENABLED");
  toogleProgramDefine(mAccumulateSamples,    "ACCUMULATE_SAMPLES");
  toogleProgramDefine(mUseUniformSampling,   "USE_UNIFORM_SAMPLING");

  uvec3 rayLaunchDims = uvec3(pCombined->getWidth(), pCombined->getHeight(), 1);
  mTracer.pSceneRenderer->renderScene(pRenderContext, mTracer.pVars, mpState, rayLaunchDims, mpScene->getActiveCamera().get());

  if (mAccumulateSamples) mNumAccumulatedSamples++;

  mLastCameraMatrix = mpScene->getActiveCamera()->getViewMatrix();
  mFrameCount++;
}
