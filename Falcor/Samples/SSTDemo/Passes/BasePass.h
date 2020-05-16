#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"
#include "Common.h"
#include "PassData.h"

using namespace Falcor;

class BasePass
{
public:
  using SharedPtr = std::shared_ptr<BasePass>;

  virtual void onFrameRender(RenderContext* pRenderContext, PassData& passData) {}
  virtual void onLoad(RenderContext* pRenderContext, PassData& passData) {}
  virtual void onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData) {}
  virtual bool onKeyEvent(const KeyboardEvent& keyEvent) { return false; }
  virtual bool onMouseEvent(const MouseEvent& mouseEvent) { return false; }
  virtual void onDataReload() {}
  virtual void onGuiRender(Gui* pGui) {}
  virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) {}
  virtual std::string getDesc() = 0;
};
