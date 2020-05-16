
// A big thanks to Toru Niina: https://github.com/ToruNiina/lbvh

#include "VPLTree.h"
#include "../Shared/VPLTreeStructs.h"
#include "../Shared/VPLData.h"

const char* VPLTree::kDesc = "VPL Tree (SST)";

namespace
{
    const char kInitShaderFile[]          = "Passes/VPLTree/TreeInit.cs.slang";
    const char kCodeShaderFile[]          = "Passes/VPLTree/TreeCode.cs.slang";
    const char kAssignLeafIdxShaderFile[] = "Passes/VPLTree/TreeAssignLeafIndex.cs.slang";
    const char kInternalNodesShaderFile[] = "Passes/VPLTree/TreeInternalNodes.cs.slang";
    const char kMergeNodesShaderFile[]    = "Passes/VPLTree/TreeMergeNodes.cs.slang";
}

VPLTree::SharedPtr VPLTree::create()
{
    SharedPtr pPass = SharedPtr(new VPLTree);
    return pPass;
}

VPLTree::VPLTree()
{
    createPrograms();
    mpBitonicSort = BitonicSort::create();
}

void VPLTree::onDataReload()
{
    createPrograms();
}

void VPLTree::onLoad(RenderContext* pRenderContext, PassData& passData)
{
}

void VPLTree::onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData)
{
}

void VPLTree::createPrograms()
{
    mpComputeState = ComputeState::create();

    mProgramDefineList.add("MAX_VPLS", "0");
    mProgramDefineList.add("NUM_SPHERE_SECTIONS", "0");

    mProgramDefineList.add("NUM_ID_BITS", "0");
    mProgramDefineList.add("NUM_DIR_BITS", "0");
    mProgramDefineList.add("NUM_MORTON_BITS", "0");

    mProgramDefineList.add("BEGIN_ID_BITS", "0");
    mProgramDefineList.add("BEGIN_DIR_BITS", "0");
    mProgramDefineList.add("BEGIN_MORTON_BITS", "0");

    const std::string SM = "6_0";
    mpInitProgram            = ComputeProgram::createFromFile(kInitShaderFile, "treeInit", mProgramDefineList, Shader::CompilerFlags::None, SM);
    mpCodeProgram            = ComputeProgram::createFromFile(kCodeShaderFile, "treeCode", mProgramDefineList, Shader::CompilerFlags::None, SM);
    mpAssignLeafIndexProgram = ComputeProgram::createFromFile(kAssignLeafIdxShaderFile, "treeAssignLeafIndex", mProgramDefineList, Shader::CompilerFlags::None, SM);
    mpInternalNodesProgram   = ComputeProgram::createFromFile(kInternalNodesShaderFile, "treeInternalNodes", mProgramDefineList, Shader::CompilerFlags::None, SM);
    mpMergeNodesProgram      = ComputeProgram::createFromFile(kMergeNodesShaderFile, "treeMergeNodes", mProgramDefineList, Shader::CompilerFlags::None, SM);

    mpInitVars            = ComputeVars::create(mpInitProgram->getReflector());
    mpCodeVars            = ComputeVars::create(mpCodeProgram->getReflector());
    mpAssignLeafIndexVars = ComputeVars::create(mpAssignLeafIndexProgram->getReflector());
    mpInternalNodesVars   = ComputeVars::create(mpInternalNodesProgram->getReflector());
    mpMergeNodesVars      = ComputeVars::create(mpMergeNodesProgram->getReflector());
}

void VPLTree::createResources(const int maxVPLs)
{
    if (mBufferMaxVPLs == maxVPLs)
        return;

    const int numTotalNodes  = getNumTotalNodes(maxVPLs);

    auto bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess;
    mpBufferNodes  = StructuredBuffer::create(mpInitProgram, "gNodes", numTotalNodes, bindFlags);
    mpBufferMerge  = StructuredBuffer::create(mpInitProgram, "gMerge", numTotalNodes, bindFlags);
    mpBufferCodes  = StructuredBuffer::create(mpCodeProgram, "gCodes", maxVPLs, bindFlags);

    mBufferMaxVPLs = maxVPLs;
}

void VPLTree::onFrameRender(RenderContext* pRenderContext, PassData& passData)
{
    PROFILE("VPLTree")

    if (!mpScene)
        return;

    int VPLUpdate = passData.getVariable<int>("VPLUpdate");

    if (VPLUpdate == 0)
        return;
    
    const int maxVPLs = passData.getVariable<int>("maxVPLs");
    const int numInternalNodes = getNumInternalNodes(maxVPLs);
    const int numTotalNodes    = getNumTotalNodes(maxVPLs);

    if (maxVPLs <= 0 || !mUpdateTree)
        return;

    StructuredBuffer::SharedPtr pBufferVPLData      = asStructuredBuffer(passData["gVPLData"]);
    StructuredBuffer::SharedPtr pBufferVPLPositions = asStructuredBuffer(passData["gVPLPositions"]);
    StructuredBuffer::SharedPtr pBufferVPLStats     = asStructuredBuffer(passData["gVPLStats"]);

    createResources(maxVPLs);

    mNumBitsDirCode = numDirCodeBits(mNumSphereSections);
    mNumBitsIdCode  = 64 - (mNumBitsMortonCode + mNumBitsDirCode);
    assert((mNumBitsMortonCode + mNumBitsDirCode + mNumBitsIdCode) == 64);

    // Compute the offsets of our 64bit code fields: [Morton|Direction|Id]
    mBeginIdCode     = 0;
    mBeginDirCode    = mNumBitsIdCode;
    mBeginMortonCode = mNumBitsDirCode + mNumBitsIdCode;
    assert(64 > mBeginMortonCode && mBeginMortonCode > mBeginDirCode && mBeginDirCode > mBeginIdCode  && mBeginIdCode == 0);

    const uint64_t numMaxSupportedVPLs = (1llu << mNumBitsIdCode) - 1;
    assert(maxVPLs <= numMaxSupportedVPLs);

    // Set shader defines
    mProgramDefineList.add("MAX_VPLS",            std::to_string(maxVPLs));
    mProgramDefineList.add("NUM_SPHERE_SECTIONS", std::to_string(mNumSphereSections));

    mProgramDefineList.add("NUM_ID_BITS",     std::to_string(mNumBitsIdCode));
    mProgramDefineList.add("NUM_DIR_BITS",    std::to_string(mNumBitsDirCode));
    mProgramDefineList.add("NUM_MORTON_BITS", std::to_string(mNumBitsMortonCode));

    mProgramDefineList.add("BEGIN_ID_BITS",    std::to_string(mBeginIdCode));
    mProgramDefineList.add("BEGIN_DIR_BITS",    std::to_string(mBeginDirCode));
    mProgramDefineList.add("BEGIN_MORTON_BITS", std::to_string(mBeginMortonCode));

    if (mShowStats)
        mVPLStats = readBuffer<VPLStats>(pBufferVPLStats)[0];

    // Dispatch buffer initialization.
    {
        PROFILE("Init");
        mpInitProgram->addDefines(mProgramDefineList);

        mpInitVars->setStructuredBuffer("gNodes", mpBufferNodes);
        mpInitVars->setStructuredBuffer("gMerge", mpBufferMerge);

        const glm::uvec3 numGroups = div_round_up(glm::uvec3(numTotalNodes, 1u, 1u), mpInitProgram->getReflector()->getThreadGroupSize());

        mpComputeState->setProgram(mpInitProgram);
        pRenderContext->setComputeState(mpComputeState);
        pRenderContext->setComputeVars(mpInitVars);
        pRenderContext->dispatch(numGroups.x, numGroups.y, numGroups.z);

        pRenderContext->uavBarrier(mpBufferNodes.get());
        pRenderContext->uavBarrier(mpBufferMerge.get());
    }

    // Dispatch compute codes
    {
        PROFILE("ComputeCodes");
        mpCodeProgram->addDefines(mProgramDefineList);

        mpCodeVars->setStructuredBuffer("gVPLData", pBufferVPLData);
        mpCodeVars->setStructuredBuffer("gCodes", mpBufferCodes);

        mpCodeVars["CB"]["gMinExtent"] = mpScene->getBoundingBox().getMinPos();
        mpCodeVars["CB"]["gMaxExtent"] = mpScene->getBoundingBox().getMaxPos();

        mpComputeState->setProgram(mpCodeProgram);
        const glm::uvec3 numGroups = div_round_up(glm::uvec3(maxVPLs, 1u, 1u), mpCodeProgram->getReflector()->getThreadGroupSize());

        mpComputeState->setProgram(mpCodeProgram);
        pRenderContext->setComputeState(mpComputeState);
        pRenderContext->setComputeVars(mpCodeVars);
        pRenderContext->dispatch(numGroups.x, numGroups.y, numGroups.z);

        pRenderContext->uavBarrier(mpBufferCodes.get());
    }

    // Dispatch code sorting
    {
        PROFILE("SortCodes");
        mpBitonicSort->execute(pRenderContext, mpBufferCodes, maxVPLs, uint2(63, 0), 128, 256);

        if (mCheckCodesSorted)
            mCodesAreSorted = checkCodesSorted(mpBufferCodes);
    }

    // Dispatch assign vpl index to nodes
    {
        PROFILE("AssignLeafIndex");
        mpAssignLeafIndexProgram->addDefines(mProgramDefineList);

        mpAssignLeafIndexVars->setStructuredBuffer("gCodes", mpBufferCodes);
        mpAssignLeafIndexVars->setStructuredBuffer("gNodes", mpBufferNodes);

        const glm::uvec3 numGroups = div_round_up(glm::uvec3(maxVPLs, 1u, 1u), mpAssignLeafIndexProgram->getReflector()->getThreadGroupSize());

        mpComputeState->setProgram(mpAssignLeafIndexProgram);
        pRenderContext->setComputeState(mpComputeState);
        pRenderContext->setComputeVars(mpAssignLeafIndexVars);
        pRenderContext->dispatch(numGroups.x, numGroups.y, numGroups.z);

        pRenderContext->uavBarrier(mpBufferNodes.get());
    }

    // Dispatch construct internal nodes
    {
        PROFILE("InternalNodes");
        mpInternalNodesProgram->addDefines(mProgramDefineList);

        mpInternalNodesVars->setStructuredBuffer("gVPLStats", pBufferVPLStats);
        mpInternalNodesVars->setStructuredBuffer("gCodes", mpBufferCodes);
        mpInternalNodesVars->setStructuredBuffer("gNodes", mpBufferNodes);

        const glm::uvec3 numGroups = div_round_up(glm::uvec3(numInternalNodes, 1u, 1u), mpInternalNodesProgram->getReflector()->getThreadGroupSize());

        mpComputeState->setProgram(mpInternalNodesProgram);
        pRenderContext->setComputeState(mpComputeState);
        pRenderContext->setComputeVars(mpInternalNodesVars);
        pRenderContext->dispatch(numGroups.x, numGroups.y, numGroups.z);

        pRenderContext->uavBarrier(mpBufferNodes.get());
    }

    // Dispatch merge nodes
    {
        PROFILE("MergeNodes");
        mpMergeNodesProgram->addDefines(mProgramDefineList);

        mpMergeNodesVars->setStructuredBuffer("gNodes", mpBufferNodes);
        mpMergeNodesVars->setStructuredBuffer("gVPLData", pBufferVPLData);
        mpMergeNodesVars->setStructuredBuffer("gMerge", mpBufferMerge);

        const auto offset = mpMergeNodesVars["CB"]["gApproxParams"].getOffset();
        ConstantBuffer::SharedPtr pCB = mpMergeNodesVars->getConstantBuffer("CB");
        pCB->setBlob(&mApproximationParameters, offset, sizeof(TreeApproxParams));

        const glm::uvec3 numGroups = div_round_up(glm::uvec3(maxVPLs, 1u, 1u), mpMergeNodesProgram->getReflector()->getThreadGroupSize());

        mpComputeState->setProgram(mpMergeNodesProgram);
        pRenderContext->setComputeState(mpComputeState);
        pRenderContext->setComputeVars(mpMergeNodesVars);
        pRenderContext->dispatch(numGroups.x, numGroups.y, numGroups.z);

        pRenderContext->uavBarrier(pBufferVPLData.get());

        if (mCheckTree)
            mTreeIsValid = checkTree(maxVPLs, pBufferVPLData);
    }

    passData.getVariable<int>("VPLUpdate") = 0;
}

void VPLTree::onGuiRender(Gui* pGui)
{
    pGui->addCheckBox("Update tree", mUpdateTree);

    pGui->addText("Approximation Parameters");

    pGui->addFloatVar("min normal score", mApproximationParameters.minNormalScore, 0.f, 1.f);
    pGui->addFloatVar("max normal z std", mApproximationParameters.maxNormalZStd, 0.f, 1.f);

    pGui->addCheckBox("Show VPL stats", mShowStats);
    if (mShowStats)
    {
        pGui->addText(("Num VPLs  = " + std::to_string(mVPLStats.numVPLs)).c_str());
        pGui->addText(("Num Paths = " + std::to_string(mVPLStats.numPaths)).c_str());
    }
    pGui->addCheckBox("Check tree", mCheckTree);
    pGui->addTooltip("Verifies the SST (VERY SLOW!)", true);
    if (mCheckTree)
    {
        pGui->addText(mTreeIsValid ? "Valid" : "Invalid", true);
    }
    pGui->addCheckBox("Check codes sorted", mCheckCodesSorted);
    pGui->addTooltip("Simply checks if the codes are sorted (SLOW!)", true);
    if (mCheckCodesSorted)
    {
        pGui->addText(mCodesAreSorted ? "Valid" : "Invalid", true);
    }
}

void VPLTree::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
}
