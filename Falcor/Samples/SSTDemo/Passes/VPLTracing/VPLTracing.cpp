#include "VPLTracing.h"

const char* VPLTracing::kDesc = "VPL generation";
const uint32_t kNumMaxLightSources = 100;

namespace
{
  const char kShaderFile[]  = "Passes/VPLTracing/VPLTracing.rt.hlsl";
  const char kComputeFile[] = "Passes/VPLTracing/ResetVPLs.cs.hlsl";
}

VPLTracing::SharedPtr VPLTracing::create()
{
  VPLTracing::SharedPtr pPass(new VPLTracing());
  return pPass;
}

VPLTracing::VPLTracing()
{
    createPrograms();
}

void VPLTracing::createPrograms()
{
    if (!mpScene) return;

    // Create ray tracing program.
    RtProgram::Desc progDesc;
    progDesc.addShaderLibrary(kShaderFile);
    progDesc.setRayGen("VPLTracingRayGen");
    progDesc.addMiss(0, "VPLTraceMiss");
    progDesc.addHitGroup(0, "VPLTraceClosestHit", "");
    progDesc.addDefine("NUM_LIGHT_SOURCES", "20");
    mTracer.pProgram = RtProgram::create(progDesc);
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mpScene);

    mTracer.pState = RtState::create();
    mTracer.pState->setMaxTraceRecursionDepth(mMaxBounces + 1);
    mTracer.pState->setProgram(mTracer.pProgram);
    mTracer.pState->setMaxTraceRecursionDepth(10);

    // Create reset program
    mVPLReset.pProgram = ComputeProgram::createFromFile(kComputeFile, "resetVPLs");
    mVPLReset.pVars    = ComputeVars::create(mVPLReset.pProgram->getReflector());
    mVPLReset.pState   = ComputeState::create();
    mVPLReset.pState->setProgram(mVPLReset.pProgram);
}

void VPLTracing::createResources(PassData& passData)
{
    if (!mpScene) return;
    if (!mTracer.pProgram) return;

    mMaxVPLs = mGuiMaxVPLs;

    createPrograms();
    determineConstantBufferAddresses();

    auto bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess;
    auto pBufferVPLData      = StructuredBuffer::create(mTracer.pProgram->getHitProgram(0), "gVPLData",      mMaxVPLs * 2,  bindFlags);
    auto pBufferVPLPositions = StructuredBuffer::create(mTracer.pProgram->getHitProgram(0), "gVPLPositions", mMaxVPLs,      bindFlags);
    auto pBufferVPLStats     = StructuredBuffer::create(mTracer.pProgram->getHitProgram(0), "gVPLStats",            1,      bindFlags);

    passData.addResource("gVPLData",      pBufferVPLData);
    passData.addResource("gVPLPositions", pBufferVPLPositions);
    passData.addResource("gVPLStats",     pBufferVPLStats);

    mReloadResources = false;
}

void VPLTracing::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = std::dynamic_pointer_cast<RtScene>(pScene);
    if (mpScene)
    {
        mTracer.pSceneRenderer = RtSceneRenderer::create(mpScene);
        createPrograms();
        mReloadResources = true;
    }
}

void VPLTracing::onLoad(RenderContext* pRenderContext, PassData& passData)
{
    createPrograms();
    createResources(passData);
}

void VPLTracing::onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData)
{
}

void VPLTracing::onDataReload()
{
    createPrograms();
}

void VPLTracing::determineConstantBufferAddresses()
{
  // Get RT-Shader CB sizes and offsets for VPL-Tracing
  {
    auto globalVars = mTracer.pVars->getGlobalVars();
    auto pReflector = globalVars->getReflection();
    const ParameterBlockReflection* pBlock = pReflector->getDefaultParameterBlock().get();
    const ReflectionVar* pVar = pBlock->getResource("CB").get();
    const ReflectionType* pType = pVar->getType().get();

    msLightInfosOffset    = pType->findMember("gLightInfos[0]")->getOffset();
    msLightInfosArraySize = pType->findMember("gLightInfos")->getType()->getTotalArraySize();
  }
}

void VPLTracing::onGuiRender(Gui* pGui)
{
    pGui->addIntVar("Max #VPLs", mGuiMaxVPLs);
    pGui->addTooltip("Maximum number of VLPs the generate", true);
    if (pGui->addButton("Apply", true))
        mReloadResources = true;

    if (pGui->addIntVar("max Bounces", mMaxBounces, 0, 30))
        if (mTracer.pState) mTracer.pState->setMaxTraceRecursionDepth(mMaxBounces + 1); // IMPORTANT: must be in the range of 0 to 31
    pGui->addTooltip("Maximum number of bounces", true);
    pGui->addIntVar("min Bounces", mMinBounces);
    pGui->addTooltip("Minimum number of bounces", true);

}

uint VPLTracing::uploadSceneLightInfos(RenderContext* pRenderContext)
{
    // Determine total power of scene lighting
    float totalPower = 0.f;
    float totalSurfaceArea = 0.f;
    for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
        totalPower += mpScene->getLight(i)->getPower();
    for (uint32_t i = 0; i < mpScene->getAreaLightCount(); i++)
    {
        totalSurfaceArea += mpScene->getAreaLight(i)->getSurfaceArea();
        totalPower += mpScene->getAreaLight(i)->getPower();
    }

    // Determine light type and number of rays to shoot per scene light source
    const uint numTotalLights = mpScene->getLightCount() + mpScene->getAreaLightCount();
    mLightInfos.resize(numTotalLights);
    uint nextLightRayStartIndex = 0;
    const uint raysToLaunch = mNumPaths = mMaxBounces > 0 ? mMaxVPLs / mMaxBounces : 0;

    // Analytic Lights
    for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
    {
        auto pLight = mpScene->getLight(i);
        mLightInfos[i].type = pLight->getType();
        mLightInfos[i].index = i;
        const float powerProportion = pLight->getPower() / totalPower;
        const uint numRays = static_cast<uint> (std::floor(raysToLaunch * powerProportion));
        mLightInfos[i].rayRange.x = nextLightRayStartIndex;
        nextLightRayStartIndex += numRays;
        mLightInfos[i].rayRange.y = nextLightRayStartIndex;
    }

    // Area lights
    const uint32_t numLights = mpScene->getLightCount();
    for (uint32_t i = 0; i < mpScene->getAreaLightCount(); i++)
    {
        auto pAreaLight = mpScene->getAreaLight(i);
        mLightInfos[numLights + i].type = pAreaLight->getType();
        mLightInfos[numLights + i].index = i;
        const float powerProportion = pAreaLight->getPower() / totalPower;
        const uint numRays = static_cast<uint> (std::floor(raysToLaunch * powerProportion));
        const float surfaceAreaProportion = pAreaLight->getSurfaceArea() / totalSurfaceArea;
        mLightInfos[numLights + i].rayRange.x = nextLightRayStartIndex;
        nextLightRayStartIndex += numRays;
        mLightInfos[numLights + i].rayRange.y = nextLightRayStartIndex;
    }

    assert(mLightInfos.size() <= msLightInfosArraySize);

    ConstantBuffer::SharedPtr pCB = mTracer.pVars->getGlobalVars()->getConstantBuffer("CB");
    pCB->setBlob(mLightInfos.data(), msLightInfosOffset, sizeof(LightInfo) * numTotalLights);

    return totalPower > 0.f ? mLightInfos.at(numTotalLights - 1).rayRange.y : 0;
}

void VPLTracing::onFrameRender(RenderContext* pRenderContext, PassData& passData)
{
    PROFILE("VPL-Tracing");

    if (!mpScene)
        return;

    if (!mUpdateVPLs)
        return;

    if (mReloadResources)
        createResources(passData);

    StructuredBuffer::SharedPtr pBufferVPLData      = asStructuredBuffer(passData["gVPLData"]);
    StructuredBuffer::SharedPtr pBufferVPLPositions = asStructuredBuffer(passData["gVPLPositions"]);
    StructuredBuffer::SharedPtr pBufferVPLStats     = asStructuredBuffer(passData["gVPLStats"]);

    // Clear VPL stats
    const VPLStats zeroStats;
    pBufferVPLStats->setBlob(&zeroStats, 0, sizeof(VPLStats));

    passData.getVariable<int>("maxVPLs")  = mMaxVPLs;

    // Reset VPL data
    {
        auto pCB = mVPLReset.pVars->getConstantBuffer("CB");
        pCB["gMaxVPLs"] = (uint) mMaxVPLs;

        mVPLReset.pVars->setStructuredBuffer("gVPLData", pBufferVPLData);
        mVPLReset.pVars->setStructuredBuffer("gVPLPositions", pBufferVPLPositions);
        mVPLReset.pVars->setStructuredBuffer("gVPLStats", pBufferVPLStats);

        const glm::uvec3 numGroups = div_round_up(glm::uvec3(mMaxVPLs, 1u, 1u), mVPLReset.pProgram->getReflector()->getThreadGroupSize());
        pRenderContext->setComputeState(mVPLReset.pState);
        pRenderContext->setComputeVars(mVPLReset.pVars);
        pRenderContext->dispatch(numGroups.x, numGroups.y, numGroups.z);
    }

    // Trace VPLs
    {
        // Prepare raytracing vars
        auto globalVars = mTracer.pVars->getGlobalVars();
        ConstantBuffer::SharedPtr pCB = globalVars->getConstantBuffer("CB");
        pCB["gMinT"]          = mMinT;
        pCB["gFrameCount"]    = mFrameCount;
        pCB["gNumMaxBounces"] = mMaxBounces;
        pCB["gNumMinBounces"] = mMinBounces;
        pCB["gNumPaths"]      = mNumPaths;

        const uint raysToLaunch = uploadSceneLightInfos(pRenderContext);
        passData.getVariable<int>("numPaths") = mNumPaths;

        // Set buffers
        globalVars->setStructuredBuffer("gVPLData", pBufferVPLData);
        globalVars->setStructuredBuffer("gVPLPositions", pBufferVPLPositions);
        globalVars->setStructuredBuffer("gVPLStats", pBufferVPLStats);

        // Launch VPL tracer
        mTracer.pSceneRenderer->renderScene(pRenderContext, mTracer.pVars, mTracer.pState, uvec3(raysToLaunch, 1, 1));

        passData.getVariable<int>("VPLUpdate") = 1;
    }

    mFrameCount++;
}
