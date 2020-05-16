#pragma once

#include "Falcor.h"
#include "FalcorExperimental.h"

#include "Passes/BasePass.h"
#include "Passes/Shared/VPLData.h"
#include "Passes/Shared/VPLTreeStructs.h"
#include "Sort/BitonicSort.h"

using namespace Falcor;


class VPLTree : public BasePass
{
public:
    using SharedPtr = std::shared_ptr<VPLTree>;

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
    VPLTree();
    void createPrograms();
    void createResources(const int maxVPLs);
    bool checkCodesSorted(StructuredBuffer::SharedPtr pBufferCodes);
    bool checkTree(const int rootNodeIndex, StructuredBuffer::SharedPtr pBufferVPLData);

    // Internal state
    Scene::SharedPtr mpScene;
    unsigned int mNumSphereSections = 3;

    const unsigned int mNumBitsMortonCode = 30;
    unsigned int mNumBitsDirCode;
    unsigned int mNumBitsIdCode;

    unsigned int mBeginMortonCode;
    unsigned int mBeginDirCode;
    unsigned int mBeginIdCode = 0;

    TreeApproxParams mApproximationParameters;
    VPLStats mVPLStats;

    bool mUpdateTree       = true;
    bool mCheckCodesSorted = false;
    bool mCheckTree        = false;
    bool mShowStats        = false;

    bool mCodesAreSorted = false;
    bool mTreeIsValid    = false;

    // Tree build compute shaders
    ComputeState::SharedPtr   mpComputeState;

    ComputeProgram::SharedPtr mpInitProgram;
    ComputeVars::SharedPtr    mpInitVars;

    ComputeProgram::SharedPtr mpCodeProgram;
    ComputeVars::SharedPtr    mpCodeVars;

    ComputeProgram::SharedPtr mpAssignLeafIndexProgram;
    ComputeVars::SharedPtr    mpAssignLeafIndexVars;

    ComputeProgram::SharedPtr mpInternalNodesProgram;
    ComputeVars::SharedPtr    mpInternalNodesVars;

    ComputeProgram::SharedPtr mpMergeNodesProgram;
    ComputeVars::SharedPtr    mpMergeNodesVars;

    Program::DefineList mProgramDefineList;

    // Tree building buffers
    StructuredBuffer::SharedPtr mpBufferCodes;
    StructuredBuffer::SharedPtr mpBufferNodes;
    StructuredBuffer::SharedPtr mpBufferMerge;

    int mBufferMaxVPLs = -1;

    BitonicSort::SharedPtr mpBitonicSort;
};
