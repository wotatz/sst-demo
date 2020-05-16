#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/GBuffer/GBuffer.h"
#include "Passes/VPLTracing/VPLTracing.h"
#include "Passes/VPLTree/VPLTree.h"
#include "Passes/VPLSampling/VPLSampling.h"
#include "Passes/SVGF/SVGF.h"
#include "Passes/RDAE/Rdae.h"
#include "Passes/TemporalFilter/TemporalFilter.h"
#include "Passes/VPLVisualizer/VPLVisualizer.h"

using namespace Falcor;


class SSTDemo : public Renderer
{
public:
  using SharedPtr = std::shared_ptr<SSTDemo>;

  SSTDemo();
  ~SSTDemo();

  void onLoad(SampleCallbacks* pCallbacks, RenderContext* pRenderContext) override;
  void onFrameRender(SampleCallbacks* pCallbacks, RenderContext* pRenderContext, const std::shared_ptr<Fbo>& pTargetFbo) override;
  void onShutdown(SampleCallbacks* pSample) override;
  void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) override;
  bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) override;
  bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) override;
  void onDataReload(SampleCallbacks* pSample) override;
  void onGuiRender(SampleCallbacks* pSample, Gui* pGui) override;

private:
  bool loadScene(RenderContext* pRenderContext, const std::string& path);
  void createResources();

  // Scene
  Scene::SharedPtr mpScene;

  // Camera
  Camera::SharedPtr           mpCamera;
  ObjectPath::SharedPtr       mpCameraPath;
  FirstPersonCameraController mCameraController;

  // Passes
  PassData mPassData;
  std::string mMainOutput;
  std::string mCurrentOutput;
  uint32_t mActiveOutputIndex = 0;

  struct {
      GBuffer::SharedPtr        pGBuffer;
      VPLTracing::SharedPtr     pVPLTracing;
      VPLTree::SharedPtr        pVPLTree;
      VPLSampling::SharedPtr    pVPLSampling;
      SVGF::SharedPtr           pSVGF;
      Rdae::SharedPtr           pRdae;
      TemporalFilter::SharedPtr pTemporalFilter;
      VPLVisualizer::SharedPtr  pVPLVisualizer;
  }mPass;

  std::vector<BasePass*> mPasses;

  // TAA Pass
  class
  {
  public:
      TemporalAA::SharedPtr pTAA;
      Fbo::SharedPtr getActiveFbo() { return pTAAFbos[activeFboIndex]; }
      Fbo::SharedPtr getInactiveFbo() { return pTAAFbos[1 - activeFboIndex]; }
      void createFbos(uint32_t width, uint32_t height, const Fbo::Desc & fboDesc)
      {
          pTAAFbos[0] = FboHelper::create2D(width, height, fboDesc);
          pTAAFbos[1] = FboHelper::create2D(width, height, fboDesc);
      }

      void switchFbos() { activeFboIndex = 1 - activeFboIndex; }
      void resetFbos()
      {
          activeFboIndex = 0;
          pTAAFbos[0] = nullptr;
          pTAAFbos[1] = nullptr;
      }

      void resetFboActiveIndex() { activeFboIndex = 0; }

  private:
      Fbo::SharedPtr pTAAFbos[2];
      uint32_t activeFboIndex = 0;
  } mTAA;

  // Gui
  Gui::DropdownList mResolutions;

  // Settings
  bool mUseRdae = true;
  bool mUseTAA  = true;
};
