/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

// Implementation of the SVGF paper.  For details, please see this paper:
//       http://research.nvidia.com/publication/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A

#include "SVGF.h"
#include "Passes/GBuffer/GBufferData.h"

const char* SVGF::kDesc = "SVGF";

namespace {
	// Where is our shaders located?
    const char kPackLinearZAndNormalShader[] = "Passes/SVGF/Shaders/SVGFPackLinearZAndNormal.ps.slang";
    const char kReprojectShader[]            = "Passes/SVGF/Shaders/SVGFReproject.ps.slang";
    const char kAtrousShader[]               = "Passes/SVGF/Shaders/SVGFAtrous.ps.slang";
    const char kFilterMomentShader[]         = "Passes/SVGF/Shaders/SVGFFilterMoments.ps.slang";
    const char kFinalModulateShader[]        = "Passes/SVGF/Shaders/SVGFFinalModulate.ps.slang";

    // Input buffer names
    const char kInputBufferAlbedo[]          = "gAlbedo";
    const char kInputBufferGBufferPacked1[]  = "gPacked1"; // World normal and emissive
    const char kInputBufferColor[]           = "gCombined";
    const char kInputBufferWorldPosition[]   = "gPosW";
    const char kInputBufferPosNormalFwidth[] = "gPosNormalFwidth";
    const char kInputBufferLinearZ[]         = "gLinearZAndNormal";
    const char kInputBufferMotionVector[]    = "gMotion";

    // Output buffer name
    const char kOutputBufferFilteredImage[]  = "gFilteredSVGF";
};

SVGF::SharedPtr SVGF::create()
{
	return SharedPtr(new SVGF());
}

SVGF::SVGF()
{
    mpState = GraphicsState::create();
    createPrograms();
    assert(mpPackLinearZAndNormal && mpReprojection && mpAtrous && mpFilterMoments && mpFinalModulate);
}

void SVGF::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
}

void SVGF::onLoad(RenderContext* pRenderContext, PassData& passData)
{
    createResources(passData);
}
void SVGF::onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData)
{
    createResources(passData);
    mBuffersNeedClear = true;
}

void SVGF::onDataReload()
{
    createPrograms();
}

void SVGF::createPrograms()
{
    mpPackLinearZAndNormal = FullScreenPass::create(kPackLinearZAndNormalShader);
    mpReprojection         = FullScreenPass::create(kReprojectShader);
    mpAtrous               = FullScreenPass::create(kAtrousShader);
    mpFilterMoments        = FullScreenPass::create(kFilterMomentShader);
    mpFinalModulate        = FullScreenPass::create(kFinalModulateShader);

    mpPackLinearZAndNormalVars = GraphicsVars::create(mpPackLinearZAndNormal->getProgram()->getReflector());
    mpReprojectionVars         = GraphicsVars::create(mpReprojection->getProgram()->getReflector());
    mpAtrousVars               = GraphicsVars::create(mpAtrous->getProgram()->getReflector());
    mpFilterMomentsVars        = GraphicsVars::create(mpFilterMoments->getProgram()->getReflector());
    mpFinalModulateVars        = GraphicsVars::create(mpFinalModulate->getProgram()->getReflector());
}

void SVGF::createResources(PassData& passData)
{
    const int width  = passData.getWidth();
    const int height = passData.getHeight();

    const auto bindFlags = Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource;
    mpInternalPreviousLighting         = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1u, 1u, nullptr, bindFlags);
    mpInternalPreviousLinearZAndNormal = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1u, 1u, nullptr, bindFlags);
    mpInternalPreviousMoments          = Texture::create2D(width, height, ResourceFormat::RG32Float,   1u, 1u, nullptr, bindFlags);

    Texture::SharedPtr pFiltered = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1u, 1u, nullptr, bindFlags);
    passData.addResource(kOutputBufferFilteredImage, pFiltered);

    allocateFbos(uvec2(width, height));
}

void SVGF::allocateFbos(uvec2 dim)
{
    {
        // Screen-size FBOs with 3 MRTs: one that is RGBA32F, one that is
        // RG32F for the luminance moments, and one that is R16F.
        Fbo::Desc desc;
        desc.setSampleCount(0);
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float); // illumination
        desc.setColorTarget(1, Falcor::ResourceFormat::RG32Float);   // moments
        desc.setColorTarget(2, Falcor::ResourceFormat::R16Float);    // history length
        mpCurReprojFbo = FboHelper::create2D(dim.x, dim.y, desc);
        mpPrevReprojFbo = FboHelper::create2D(dim.x, dim.y, desc);
    }

    {
        // Screen-size RGBA32F buffer for linear Z, derivative, and packed normal
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        mpLinearZAndNormalFbo = FboHelper::create2D(dim.x, dim.y, desc);
    }

    {
        // Screen-size FBOs with 1 RGBA32F buffer
        Fbo::Desc desc;
        desc.setColorTarget(0, Falcor::ResourceFormat::RGBA32Float);
        mpPingPongFbo[0] = FboHelper::create2D(dim.x, dim.y, desc);
        mpPingPongFbo[1] = FboHelper::create2D(dim.x, dim.y, desc);
        mpFilteredPastFbo = FboHelper::create2D(dim.x, dim.y, desc);
        mpFilteredIlluminationFbo = FboHelper::create2D(dim.x, dim.y, desc);
        mpFinalFbo = FboHelper::create2D(dim.x, dim.y, desc);
    }

    mBuffersNeedClear = true;
}

void SVGF::clearBuffers(RenderContext* pRenderContext, PassData& passData)
{
    pRenderContext->clearFbo(mpPingPongFbo[0].get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpPingPongFbo[1].get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpLinearZAndNormalFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFilteredPastFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpCurReprojFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpPrevReprojFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpFilteredIlluminationFbo.get(), glm::vec4(0), 1.0f, 0, FboAttachmentType::All);

    pRenderContext->clearRtv(mpInternalPreviousLinearZAndNormal->getRTV().get(), vec4(0, 0, 0, 1));
    pRenderContext->clearRtv(mpInternalPreviousLighting->getRTV().get(), vec4(0, 0, 0, 1));
    pRenderContext->clearRtv(mpInternalPreviousMoments->getRTV().get(), vec4(0, 0, 0, 1));
}

void SVGF::onFrameRender(RenderContext* pRenderContext, PassData& passData)
{
    PROFILE("SVGF");

    Texture::SharedPtr pAlbedoTexture          = asTexture(passData[kInputBufferAlbedo]);
    Texture::SharedPtr pColorTexture           = asTexture(passData[kInputBufferColor]);
    Texture::SharedPtr pGBufferPacked1         = asTexture(passData[kInputBufferGBufferPacked1]);
    Texture::SharedPtr pWorldPositionTexture   = asTexture(passData[kInputBufferWorldPosition]);
    Texture::SharedPtr pPosNormalFwidthTexture = asTexture(passData[kInputBufferPosNormalFwidth]);
    Texture::SharedPtr pLinearZTexture         = asTexture(passData[kInputBufferLinearZ]);
    Texture::SharedPtr pMotionVectorTexture    = asTexture(passData[kInputBufferMotionVector]);
    Texture::SharedPtr pOutputTexture          = asTexture(passData[kOutputBufferFilteredImage]);

    assert(pAlbedoTexture
        && pColorTexture
        && pGBufferPacked1
        && pWorldPositionTexture
        && pPosNormalFwidthTexture
        && pLinearZTexture
        && pMotionVectorTexture
        && pOutputTexture);

    assert(mpFilteredIlluminationFbo &&
        mpFilteredIlluminationFbo->getWidth() == pAlbedoTexture->getWidth() &&
        mpFilteredIlluminationFbo->getHeight() == pAlbedoTexture->getHeight());

    if (mBuffersNeedClear)
    {
        clearBuffers(pRenderContext, passData);
        mBuffersNeedClear = false;
    }

    if (mFilterEnabled)
    {
        pRenderContext->setGraphicsState(mpState);

        // Grab linear z and its derivative and also pack the normal into
        // the last two channels of the mpLinearZAndNormalFbo.
        computeLinearZAndNormal(pRenderContext, pLinearZTexture, pGBufferPacked1);

        // Demodulate input color & albedo to get illumination and lerp in
        // reprojected filtered illumination from the previous frame.
        // Stores the result as well as initial moments and an updated
        // per-pixel history length in mpCurReprojFbo.
        computeReprojection(pRenderContext, pAlbedoTexture, pColorTexture, pGBufferPacked1, pMotionVectorTexture, pPosNormalFwidthTexture, mpInternalPreviousLinearZAndNormal);

        // Do a first cross-bilateral filtering of the illumination and
        // estimate its variance, storing the result into a float4 in
        // mpPingPongFbo[0].  Takes mpCurReprojFbo as input.
        computeFilteredMoments(pRenderContext);

        // Filter illumination from mpCurReprojFbo[0], storing the result
        // in mpPingPongFbo[0].  Along the way (or at the end, depending on
        // the value of mFeedbackTap), save the filtered illumination for
        // next time into mpFilteredPastFbo.
        computeAtrousDecomposition(pRenderContext, pAlbedoTexture);

        // Compute albedo * filtered illumination and add emission back in.
        auto perImageCB = mpFinalModulateVars["PerImageCB"];
        mpFinalModulateVars->setTexture("gAlbedo", pAlbedoTexture);
        mpFinalModulateVars->setTexture("gPacked1", pGBufferPacked1);
        mpFinalModulateVars->setTexture("gIllumination", mpPingPongFbo[0]->getColorTexture(0));

        mpState->setFbo(mpFinalFbo);
        pRenderContext->setGraphicsVars(mpFinalModulateVars);
        mpFinalModulate->execute(pRenderContext);

        // Blit into the output texture.
        pRenderContext->blit(mpFinalFbo->getColorTexture(0)->getSRV(), pOutputTexture->getRTV());

        // Swap resources so we're ready for next frame.
        std::swap(mpCurReprojFbo, mpPrevReprojFbo);
        pRenderContext->blit(mpLinearZAndNormalFbo->getColorTexture(0)->getSRV(), mpInternalPreviousLinearZAndNormal->getRTV());
    }
    else
    {
        pRenderContext->blit(pColorTexture->getSRV(), pOutputTexture->getRTV());
    }
}

// Extracts linear z and its derivative from the linear Z texture and packs
// the normal from the world normal texture and packes them into the FBO.
// (It's slightly wasteful to copy linear z here, but having this all
// together in a single buffer is a small simplification, since we make a
// copy of it to refer to in the next frame.)
void SVGF::computeLinearZAndNormal(RenderContext* pRenderContext, Texture::SharedPtr pLinearZTexture,
    Texture::SharedPtr pGBufferPacked1)
{
    auto perImageCB = mpPackLinearZAndNormalVars["PerImageCB"];
    mpPackLinearZAndNormalVars->setTexture("gLinearZ", pLinearZTexture);
    mpPackLinearZAndNormalVars->setTexture("gPacked1", pGBufferPacked1);

    mpState->setFbo(mpLinearZAndNormalFbo);
    pRenderContext->pushGraphicsState(mpState);
    pRenderContext->pushGraphicsVars(mpPackLinearZAndNormalVars);
    mpPackLinearZAndNormal->execute(pRenderContext);
    pRenderContext->popGraphicsVars();
    pRenderContext->popGraphicsState();
}

void SVGF::computeReprojection(RenderContext* pRenderContext, Texture::SharedPtr pAlbedoTexture,
    Texture::SharedPtr pColorTexture, Texture::SharedPtr pGBufferPacked1,
    Texture::SharedPtr pMotionVectorTexture,
    Texture::SharedPtr pPositionNormalFwidthTexture,
    Texture::SharedPtr pPrevLinearZTexture)
{
    auto perImageCB = mpReprojectionVars["PerImageCB"];

    // Setup textures for our reprojection shader pass
    mpReprojectionVars->setTexture("gMotion", pMotionVectorTexture);
    mpReprojectionVars->setTexture("gColor", pColorTexture);
    mpReprojectionVars->setTexture("gAlbedo", pAlbedoTexture);
    mpReprojectionVars->setTexture("gPacked1", pGBufferPacked1);
    mpReprojectionVars->setTexture("gPositionNormalFwidth", pPositionNormalFwidthTexture);
    mpReprojectionVars->setTexture("gPrevIllum", mpFilteredPastFbo->getColorTexture(0));
    mpReprojectionVars->setTexture("gPrevMoments", mpPrevReprojFbo->getColorTexture(1));
    mpReprojectionVars->setTexture("gLinearZAndNormal", mpLinearZAndNormalFbo->getColorTexture(0));
    mpReprojectionVars->setTexture("gPrevLinearZAndNormal", pPrevLinearZTexture);
    mpReprojectionVars->setTexture("gPrevHistoryLength", mpPrevReprojFbo->getColorTexture(2));

    // Setup variables for our reprojection pass
    perImageCB["gAlpha"] = mAlpha;
    perImageCB["gMomentsAlpha"] = mMomentsAlpha;

    mpState->setFbo(mpCurReprojFbo);
    pRenderContext->pushGraphicsState(mpState);
    pRenderContext->pushGraphicsVars(mpReprojectionVars);
    mpReprojection->execute(pRenderContext);
    pRenderContext->popGraphicsVars();
    pRenderContext->popGraphicsState();
}

void SVGF::computeFilteredMoments(RenderContext* pRenderContext)
{
    auto perImageCB = mpFilterMomentsVars["PerImageCB"];

    mpFilterMomentsVars->setTexture("gIllumination", mpCurReprojFbo->getColorTexture(0));
    mpFilterMomentsVars->setTexture("gHistoryLength", mpCurReprojFbo->getColorTexture(2));
    mpFilterMomentsVars->setTexture("gLinearZAndNormal", mpLinearZAndNormalFbo->getColorTexture(0));
    mpFilterMomentsVars->setTexture("gMoments", mpCurReprojFbo->getColorTexture(1));

    perImageCB["gPhiColor"] = mPhiColor;
    perImageCB["gPhiNormal"] = mPhiNormal;

    mpState->setFbo(mpPingPongFbo[0]);
    pRenderContext->setGraphicsVars(mpFilterMomentsVars);
    mpFilterMoments->execute(pRenderContext);
}

void SVGF::computeAtrousDecomposition(RenderContext* pRenderContext, Texture::SharedPtr pAlbedoTexture)
{
    auto perImageCB = mpAtrousVars["PerImageCB"];

    mpAtrousVars->setTexture("gAlbedo", pAlbedoTexture);
    mpAtrousVars->setTexture("gHistoryLength", mpCurReprojFbo->getColorTexture(2));
    mpAtrousVars->setTexture("gLinearZAndNormal", mpLinearZAndNormalFbo->getColorTexture(0));

    perImageCB["gPhiColor"] = mPhiColor;
    perImageCB["gPhiNormal"] = mPhiNormal;

    for (int i = 0; i < mFilterIterations; i++)
    {
        Fbo::SharedPtr curTargetFbo = mpPingPongFbo[1];

        mpAtrousVars->setTexture("gIllumination", mpPingPongFbo[0]->getColorTexture(0));
        perImageCB["gStepSize"] = 1 << i;

        mpState->setFbo(curTargetFbo);
        pRenderContext->setGraphicsVars(mpAtrousVars);
        mpAtrous->execute(pRenderContext);

        // store the filtered color for the feedback path
        if (i == std::min(mFeedbackTap, mFilterIterations - 1))
        {
            pRenderContext->blit(curTargetFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
        }

        std::swap(mpPingPongFbo[0], mpPingPongFbo[1]);
    }

    if (mFeedbackTap < 0)
    {
        pRenderContext->blit(mpCurReprojFbo->getColorTexture(0)->getSRV(), mpFilteredPastFbo->getRenderTargetView(0));
    }
}

void SVGF::onGuiRender(Gui* pGui)
{
    int dirty = 0;
    dirty |= (int)pGui->addCheckBox(mFilterEnabled ? "SVGF enabled" : "SVGF disabled", mFilterEnabled);

    pGui->addText("");
    pGui->addText("Number of filter iterations.  Which");
    pGui->addText("    iteration feeds into future frames?");
    dirty |= (int)pGui->addIntVar("Iterations", mFilterIterations, 2, 10, 1);
    dirty |= (int)pGui->addIntVar("Feedback", mFeedbackTap, -1, mFilterIterations - 2, 1);

    pGui->addText("");
    pGui->addText("Contol edge stopping on bilateral fitler");
    dirty |= (int)pGui->addFloatVar("For Color", mPhiColor, 0.0f, 10000.0f, 0.01f);
    dirty |= (int)pGui->addFloatVar("For Normal", mPhiNormal, 0.001f, 1000.0f, 0.2f);

    pGui->addText("");
    pGui->addText("How much history should be used?");
    pGui->addText("    (alpha; 0 = full reuse; 1 = no reuse)");
    dirty |= (int)pGui->addFloatVar("Alpha", mAlpha, 0.0f, 1.0f, 0.001f);
    dirty |= (int)pGui->addFloatVar("Moments Alpha", mMomentsAlpha, 0.0f, 1.0f, 0.001f);

    if (dirty) mBuffersNeedClear = true;
}
