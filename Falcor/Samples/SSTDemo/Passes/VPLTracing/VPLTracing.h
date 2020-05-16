#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/BasePass.h"
#include "Passes/Shared/VPLData.h"

using namespace Falcor;


class VPLTracing : public BasePass
{
public:
  using SharedPtr = std::shared_ptr<VPLTracing>;

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
  VPLTracing();
  void createPrograms();
  void createResources(PassData& passData);

  RtScene::SharedPtr mpScene;

  /** Examine scene lights and update constant buffer. Returns number of rays to launch.
  */
  uint uploadSceneLightInfos(RenderContext* pRenderContext);

  void determineConstantBufferAddresses();

  // Ray tracing program.
  struct
  {
      RtSceneRenderer::SharedPtr pSceneRenderer;
      RtProgram::SharedPtr       pProgram;
      RtProgramVars::SharedPtr   pVars;
      RtState::SharedPtr         pState;
  } mTracer;

  // VPL reset program.
  struct
  {
      ComputeProgram::SharedPtr pProgram;
      ComputeVars::SharedPtr    pVars;
      ComputeState::SharedPtr   pState;
  } mVPLReset;

  // Constant buffer infos
  size_t msLightInfosOffset = -1;
  size_t msLightInfosArraySize = -1;

  // Various internal parameters
  bool mReloadResources = false;
  bool mUpdateVPLs = true;
  bool mRecreateLightCollection = false;
  bool mUseTriangleNormals = false;
  float mMinT = 0.001f;
  int mNumPaths = 0;
  int mMaxVPLs = 100000;
  int mGuiMaxVPLs = mMaxVPLs;
  int mMaxBounces = 3;
  int mMinBounces = 1;
  uint32_t mFrameCount = 0x1337u;  // Frame counter to vary random numbers over time
  std::vector<LightInfo> mLightInfos;
};
