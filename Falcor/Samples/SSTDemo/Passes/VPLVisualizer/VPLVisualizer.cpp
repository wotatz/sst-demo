#include "VPLVisualizer.h"
#include "../Shared/VPLData.h"
#include "../Shared/VPLUtils.h"

const char* VPLVisualizer::kDesc = "Visualization";

namespace
{
    const char kVPLVisualizerShaderFile[] = "Passes/VPLVisualizer/VPLVisualizer.slang";

    void drawPlaneVariance(DebugDrawer::SharedPtr pDebugDrawer, const float3& P, const float3& N, const float3& V)
    {
        const float3x3 R = getRotationMatrixFromAToB(N, float3(0.f, 0.f, 1.f));
        const float3 origin = P + N * 0.001f; // Offset origin a little bit so that lines are in front of geometry

        pDebugDrawer->setColor(float3(1.f, 0.f, 0.f)); // Red
        pDebugDrawer->addLine(origin, origin + R[0] * std::sqrt(V.x));
        pDebugDrawer->setColor(float3(0.f, 1.f, 0.f)); // Green
        pDebugDrawer->addLine(origin, origin + R[1] * std::sqrt(V.y));
        pDebugDrawer->setColor(float3(0.f, 0.f, 1.f)); // Blue
        pDebugDrawer->addLine(origin, origin + R[2] * std::sqrt(V.z));
    }
}

VPLVisualizer::SharedPtr VPLVisualizer::create()
{
    SharedPtr pPass = SharedPtr(new VPLVisualizer);
    return pPass;
}

VPLVisualizer::VPLVisualizer()
{
    // Setup VPL renderer
    mpVPLModel = Model::createFromFile("Data/pl_diamond.obj");

    mpFbo = Fbo::create();
    mpState = GraphicsState::create();

    RasterizerState::Desc rasterizerDesc;
    rasterizerDesc.setCullMode(RasterizerState::CullMode::Back);
    mpState->setRasterizerState(RasterizerState::create(rasterizerDesc));

    DepthStencilState::Desc dssState;
    dssState.setDepthTest(true).setStencilTest(false);
    mpState->setDepthStencilState(DepthStencilState::create(dssState));

    createPrograms();

    // Setup Debug Drawer
    mpDebugDrawer = DebugDrawer::create();

    mpDebugDrawerProgram = GraphicsProgram::createFromFile("Passes/Shared/DebugDrawer.slang", "debugDrawVs", "debugDrawPs");
    mpDebugDrawerVars = GraphicsVars::create(mpDebugDrawerProgram->getReflector());

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthTest(true).setStencilTest(false);
    DepthStencilState::SharedPtr depthTestDs = DepthStencilState::create(dsDesc);

    RasterizerState::Desc lineRSDesc;
    lineRSDesc.setFillMode(RasterizerState::FillMode::Solid).setCullMode(RasterizerState::CullMode::None);

    mpDebugDrawerFbo = Fbo::create();
    mpDebugDrawerGraphicsState = GraphicsState::create();
    mpDebugDrawerGraphicsState->setFbo(mpDebugDrawerFbo);
    mpDebugDrawerGraphicsState->setProgram(mpDebugDrawerProgram);
    mpDebugDrawerGraphicsState->setDepthStencilState(depthTestDs);
    mpDebugDrawerGraphicsState->setRasterizerState(RasterizerState::create(lineRSDesc));
}

void VPLVisualizer::onLoad(RenderContext* pRenderContext, PassData& passData)
{
}

void VPLVisualizer::onResizeSwapChain(uint32_t width, uint32_t height, PassData& passData)
{
}

void VPLVisualizer::onDataReload()
{
    createPrograms();
}

void VPLVisualizer::createPrograms()
{
    mVPLRenderer.pProgram = GraphicsProgram::createFromFile(kVPLVisualizerShaderFile, "vs", "ps");
    mVPLRenderer.pVars.reset();
    mVPLRenderer.pVars = GraphicsVars::create(mVPLRenderer.pProgram->getReflector());
}

void VPLVisualizer::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    // Set new scene.
    mpScene = pScene;

    if (pScene)
    {
        createPrograms();
    }
}

void VPLVisualizer::onGuiRender(Gui* pGui)
{
    pGui->addSeparator();
    pGui->addFloatVar("VPLRenderScale", mVPLRenderScale, 0.1f, 10.f);
    pGui->addFloatVar("ColorDenom", mVPLColorDenom, 0.1f, 100.f);
    pGui->addSeparator();

    pGui->addCheckBox("Enable VPL visualization", mEnableVPLVisualization);
    pGui->addCheckBox("Enable Tree visualization", mEnableVPLTreeVisualization);

    if (mEnableVPLTreeVisualization)
    {
        pGui->addCheckBox("Show variance", mShowVPLVariance);
        pGui->addSeparator();
        pGui->addCheckBox("Draw Approx", mShowVPLTreeApprox);
        pGui->addTooltip("Renders tree nodes suitable for earyl stopping", true);
        if (!mShowVPLTreeApprox)
            pGui->addIntVar("Level to draw", mVPLTreeLevel, 0, 100, 1);
        else
            pGui->addCheckBox("Use VPL color", mVPLTreeUseColor);
    }
}

void VPLVisualizer::onFrameRender(RenderContext* pRenderContext, PassData& passData)
{
    PROFILE("VPLVisualizer");

    if (!mpScene)
        return;

    Texture::SharedPtr pColor = asTexture(passData["gColor"]);
    Texture::SharedPtr pDepth = asTexture(passData["gDepth"]);

    if (mEnableVPLVisualization)
    {
        StructuredBuffer::SharedPtr pBufferVPLData = asStructuredBuffer(passData["gVPLData"]);
        StructuredBuffer::SharedPtr pBufferVPLStats = asStructuredBuffer(passData["gVPLStats"]);

        mVPLRenderer.pVars->setStructuredBuffer("gVPLData", pBufferVPLData);
        mVPLRenderer.pVars->setStructuredBuffer("gVPLStats", pBufferVPLStats);

        mVPLRenderer.pVars["CB"]["gCamViewProjMat"] = mpScene->getActiveCamera()->getViewProjMatrix();
        mVPLRenderer.pVars["CB"]["gVPLRenderScale"] = mVPLRenderScale;
        mVPLRenderer.pVars["CB"]["gVPLColorScale"]  = mVPLColorDenom;

        VPLStats stats;
        pBufferVPLStats->readBlob(&stats, 0, sizeof(VPLStats));

        int maxVPLs = passData.getVariable<int>("maxVPLs");

        mpFbo->attachColorTarget(pColor, 0);
        mpFbo->attachDepthStencilTarget(pDepth);

        const auto mesh = mpVPLModel->getMesh(0);
        mpState->setVao(mesh->getVao());
        mpState->setFbo(mpFbo);
        mpState->setProgram(mVPLRenderer.pProgram);

        pRenderContext->pushGraphicsState(mpState);
        pRenderContext->pushGraphicsVars(mVPLRenderer.pVars);
        pRenderContext->drawIndexedInstanced(mesh->getIndexCount(), maxVPLs, 0, 0, 0);
        pRenderContext->popGraphicsVars();
        pRenderContext->popGraphicsState();
    }

    if (mEnableVPLTreeVisualization)
    {
        renderVPLTree(pRenderContext, passData, pColor);
    }
}

void VPLVisualizer::renderVPLTree(RenderContext* pRenderContext, PassData& passData, Texture::SharedPtr pTarget)
{
    const float minBBSize = 0.001f;
    mpDebugDrawer->clear();

    StructuredBuffer::SharedPtr pBufferVPLData  = asStructuredBuffer(passData["gVPLData"]);
    StructuredBuffer::SharedPtr pBufferVPLStats = asStructuredBuffer(passData["gVPLStats"]);

    std::vector<VPLData> vplTree = readBuffer<VPLData>(pBufferVPLData);
    int maxVPLs   = passData.getVariable<int>("maxVPLs");
    int numPaths  = passData.getVariable<int>("numPaths");
    int rootIndex = maxVPLs;

    const std::function<void(const std::vector<VPLData>&, int)> collectApprox = [&](const std::vector<VPLData>& vplTreeData, int index) -> void
    {
        if (index >= 0 && index < vplTreeData.size())
        {
            const VPLData&    vplData = vplTreeData.at(index);

            if (vplData.getEarlyStop() <= 0.f && vplData.numVPLSubTree > 0)
            {
                collectApprox(vplTreeData, vplData.idChild1);
                collectApprox(vplTreeData, vplData.idChild2);
            }
            else
            {
                auto bb = BoundingBox::fromMinMax(vplData.getAABBMin(), vplData.getAABBMax());
                bb.extent = max(bb.extent, minBBSize);

                if (mVPLTreeUseColor)
                    mpDebugDrawer->setColor(vplData.getColor() * (float)(numPaths) / (float)vplData.numVPLSubTree / mVPLColorDenom);
                else
                    mpDebugDrawer->setColor(float3(0.35f, 0.8, 0.35f));

               mpDebugDrawer->addBoundingBox(bb);

                // Draw variance 
                if (mShowVPLVariance)
                {
                    drawPlaneVariance(mpDebugDrawer, vplData.getPosW(), vplData.getNormW(), vplData.numVPLSubTree > 0 ? vplData.getVariance() : float3(0.f));
                }
            }
        }
    };

    const std::function<void(const std::vector<VPLData>&, int, int, int)> collectLevels = [&, this](const std::vector<VPLData>& vplTreeData, int index, int currentLevel, int drawLevel) -> void
    {
        if (index >= 0 && index < vplTreeData.size())
        {
            const VPLData& vplData = vplTreeData.at(index);
            if (currentLevel == drawLevel)
            {
                // Draw AABB
                auto bb = BoundingBox::fromMinMax(vplData.getAABBMin(), vplData.getAABBMax());
                bb.extent = max(bb.extent, minBBSize);

                const float3 drawColor = vplData.getColor() * (float)(numPaths) / (float)vplData.numVPLSubTree / mVPLColorDenom;

                mpDebugDrawer->setColor(drawColor);
                mpDebugDrawer->addBoundingBox(bb);

                // Draw approximated position
                BoundingBox approxPos;
                approxPos.center = vplData.getPosW();
                approxPos.extent = float3(0.001f);
                mpDebugDrawer->setColor(drawColor);
                mpDebugDrawer->addBoundingBox(approxPos);

                // Draw approximated normal
                if (!mShowVPLVariance)
                {
                    mpDebugDrawer->setColor(float3(0.4f, 1.f, 0.9f));
                    mpDebugDrawer->addLine(vplData.getPosW(), vplData.getPosW() + vplData.getNormW() * 0.01f);
                }
                else // Draw variance 
                {
                    drawPlaneVariance(mpDebugDrawer, vplData.getPosW(), vplData.getNormW(), vplData.getVariance());
                }
            }
            else
            {
                collectLevels(vplTreeData, vplData.idChild1, currentLevel + 1, drawLevel);
                collectLevels(vplTreeData, vplData.idChild2, currentLevel + 1, drawLevel);
            }
        }
    };

    if (mShowVPLTreeApprox)
        collectApprox(vplTree, rootIndex);
    else
        collectLevels(vplTree, rootIndex, 0, mVPLTreeLevel);

    Texture::SharedPtr pDepth = asTexture(passData["gDepth"]);
    const auto& pCamera = mpScene->getActiveCamera();

    mpDebugDrawerFbo->attachDepthStencilTarget(pDepth);
    mpDebugDrawerFbo->attachColorTarget(pTarget, 0);
    pRenderContext->setGraphicsState(mpDebugDrawerGraphicsState);
    pRenderContext->getGraphicsState()->setFbo(mpDebugDrawerFbo);
    pRenderContext->setGraphicsVars(mpDebugDrawerVars);
    mpDebugDrawer->render(pRenderContext, pCamera.get());
}
