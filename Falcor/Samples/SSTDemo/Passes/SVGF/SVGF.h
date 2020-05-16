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

#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/BasePass.h"
#include "Passes/Shared/VPLData.h"

using namespace Falcor;


class SVGF : public BasePass
{
public:
    using SharedPtr = std::shared_ptr<SVGF>;

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
    SVGF();
    void createPrograms();
    void createResources(PassData& passData);

    void allocateFbos(uvec2 dim);
    void clearBuffers(RenderContext* pRenderContext, PassData& passData);

    void computeLinearZAndNormal(RenderContext* pRenderContext, Texture::SharedPtr pLinearZTexture, Texture::SharedPtr pGBufferPacked1);
    void computeReprojection(RenderContext* pRenderContext, Texture::SharedPtr pAlbedoTexture, Texture::SharedPtr pColorTexture, Texture::SharedPtr pGBufferPacked1,
        Texture::SharedPtr pMotionVectorTexture, Texture::SharedPtr pPositionNormalFwidthTexture, Texture::SharedPtr pPrevLinearZAndNormalTexture);
    void computeFilteredMoments(RenderContext* pRenderContext);
    void computeAtrousDecomposition(RenderContext* pRenderContext, Texture::SharedPtr pAlbedoTexture);

    bool mBuffersNeedClear = false;

    // SVGF parameters
    bool    mFilterEnabled = true;
    int32_t mFilterIterations = 4;
    int32_t mFeedbackTap = 1;
    float   mVarainceEpsilon = 1e-4f;
    float   mPhiColor = 2.0f;
    float   mPhiNormal = 128.0f;
    float   mAlpha = 0.05f;
    float   mMomentsAlpha = 0.2f;

    // SVGF passes
    FullScreenPass::UniquePtr mpPackLinearZAndNormal;
    FullScreenPass::UniquePtr mpReprojection;
    FullScreenPass::UniquePtr mpFilterMoments;
    FullScreenPass::UniquePtr mpAtrous;
    FullScreenPass::UniquePtr mpFinalModulate;

    GraphicsVars::SharedPtr mpPackLinearZAndNormalVars;
    GraphicsVars::SharedPtr mpReprojectionVars;
    GraphicsVars::SharedPtr mpFilterMomentsVars;
    GraphicsVars::SharedPtr mpAtrousVars;
    GraphicsVars::SharedPtr mpFinalModulateVars;

    GraphicsState::SharedPtr  mpState;

    // Intermediate framebuffers
    Fbo::SharedPtr mpPingPongFbo[2];
    Fbo::SharedPtr mpLinearZAndNormalFbo;
    Fbo::SharedPtr mpFilteredPastFbo;
    Fbo::SharedPtr mpCurReprojFbo;
    Fbo::SharedPtr mpPrevReprojFbo;
    Fbo::SharedPtr mpFilteredIlluminationFbo;
    Fbo::SharedPtr mpFinalFbo;

    // Internal textures
    Texture::SharedPtr mpInternalPreviousLinearZAndNormal;
    Texture::SharedPtr mpInternalPreviousLighting;
    Texture::SharedPtr mpInternalPreviousMoments;
};
