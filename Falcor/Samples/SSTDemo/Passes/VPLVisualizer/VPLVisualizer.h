#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/BasePass.h"
#include "Passes/Shared/VPLData.h"

using namespace Falcor;


class VPLVisualizer : public BasePass
{
public:
  using SharedPtr = std::shared_ptr<VPLVisualizer>;

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
    VPLVisualizer();

  void createPrograms();
  void renderVPLTree(RenderContext* pRenderContext, PassData& passData, Texture::SharedPtr pTarget);

  // VPL renderer
  struct
  {
      GraphicsProgram::SharedPtr  pProgram;
      GraphicsVars::SharedPtr     pVars; 
  } mVPLRenderer;

  GraphicsState::SharedPtr    mpState;
  Fbo::SharedPtr              mpFbo;

  Scene::SharedPtr mpScene;

  // VPL tree debug drawer
  Fbo::SharedPtr             mpDebugDrawerFbo;
  DebugDrawer::SharedPtr     mpDebugDrawer;
  GraphicsState::SharedPtr   mpDebugDrawerGraphicsState;
  GraphicsProgram::SharedPtr mpDebugDrawerProgram;
  GraphicsVars::SharedPtr    mpDebugDrawerVars;

  // Visualization settings
  bool  mEnableVPLVisualization = false;
  float mVPLRenderScale = 1.f;
  float mVPLColorDenom  = 1.f;

  bool  mEnableVPLTreeVisualization = false;
  bool  mShowVPLVariance   = true;
  bool  mVPLTreeUseColor   = false;
  bool  mShowVPLTreeApprox = true;
  int   mVPLTreeLevel = 0;

  Model::SharedPtr mpVPLModel;
};
