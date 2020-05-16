#include "SSTDemo.h"
#include "passes/gbuffer/GBufferData.h"

#include <dear_imgui/imgui.h>


namespace
{
    Gui::DropdownList kDropDownOutputs;

    Gui::DropdownList createDropdownFromVec(const std::vector<std::string>& strVec, const std::string& currentLabel)
    {
        Gui::DropdownList dropdown;
        for (size_t i = 0; i < strVec.size(); i++)
            dropdown.push_back({ (uint32_t)i, strVec[i] });
        return dropdown;
    }

    bool isInVector(const std::vector<std::string>& strVec, const std::string& str)
    {
        return std::find(strVec.begin(), strVec.end(), str) != strVec.end();
    }

    enum class EnumResolution : uint32_t { _720p = 0, _1080p, _1440p, MAX };
    std::array<ivec2, (size_t)EnumResolution::MAX> kResolutions = { ivec2(1280, 720), ivec2(1920, 1080), ivec2(2560, 1440) };
}

SSTDemo::SSTDemo()
{
    mResolutions.push_back(Gui::DropdownValue({ (uint32_t)EnumResolution::_720p,  "720p" }));
    mResolutions.push_back(Gui::DropdownValue({ (uint32_t)EnumResolution::_1080p, "1080p" }));
    mResolutions.push_back(Gui::DropdownValue({ (uint32_t)EnumResolution::_1440p, "1440p" }));
}

SSTDemo::~SSTDemo()
{
}

void SSTDemo::onShutdown(SampleCallbacks* pSample)
{
}

void SSTDemo::onLoad(SampleCallbacks* pCallbacks, RenderContext* pRenderContext)
{
    mPassData.setWidth(pCallbacks->getWindow()->getClientAreaWidth());
    mPassData.setHeight(pCallbacks->getWindow()->getClientAreaHeight());

    createResources();

    mPass.pGBuffer        = GBuffer::create();
    mPass.pVPLTracing     = VPLTracing::create();
    mPass.pVPLTree        = VPLTree::create();
    mPass.pVPLSampling    = VPLSampling::create();
    mPass.pSVGF           = SVGF::create();
    mPass.pRdae           = Rdae::create();
    mPass.pTemporalFilter = TemporalFilter::create();
    mPass.pVPLVisualizer  = VPLVisualizer::create();

    mPasses.push_back(mPass.pGBuffer.get());
    mPasses.push_back(mPass.pVPLTracing.get());
    mPasses.push_back(mPass.pVPLTree.get());
    mPasses.push_back(mPass.pVPLSampling.get());
    mPasses.push_back(mPass.pSVGF.get());
    mPasses.push_back(mPass.pRdae.get());
    mPasses.push_back(mPass.pTemporalFilter.get());
    mPasses.push_back(mPass.pVPLVisualizer.get());

    mTAA.pTAA = TemporalAA::create();

    mTAA.pTAA->setAlphaValue(0.3f);

    for (auto& pPass : mPasses)
        pPass->onLoad(pRenderContext, mPassData);

    //loadScene(pRenderContext, "Scenes/sponza/sponza_animated_teapot.fscene");
    loadScene(pRenderContext, "../Scenes/sponza.fscene");

    // Prepare output dropdown list
    std::vector<std::string> outputs;
    for (const auto& pair : mPassData.getResourceMap())
        outputs.push_back(pair.first);

    kDropDownOutputs = createDropdownFromVec(outputs, mCurrentOutput);

    // Find main output
    mMainOutput = "gColor";
    auto it = std::find_if(kDropDownOutputs.begin(), kDropDownOutputs.end(), [&](const auto& val) { return val.label == mMainOutput; });
    mActiveOutputIndex = (it != kDropDownOutputs.end()) ? (uint32_t)std::distance(kDropDownOutputs.begin(), it) : 0u;
}

void SSTDemo::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    // Scene Gui
    if (pGui->beginGroup("Scene", false))
    {
        if (pGui->addButton("Load Scene"))
        {
            std::string filename;
            if (openFileDialog(Scene::kFileExtensionFilters, filename))
            {
                ProgressBar::SharedPtr pBar = ProgressBar::create("Loading Scene", 100);
                loadScene(pSample->getRenderContext(), filename);
            }
        }
        if (mpScene)
        {
            mpScene->renderUI(pGui);
        }
        pGui->endGroup();
    }

    pGui->addSeparator();

    // Output to render
    pGui->addText("Output");
    pGui->addTooltip("Output intermediate textures", true);
    pGui->addDropdown("##Output", kDropDownOutputs, mActiveOutputIndex, true);
    mCurrentOutput = kDropDownOutputs.at(mActiveOutputIndex).label;

    // Window resolution
    static uint32_t CurrentResolution = 0;
    pGui->addText("Resolution");
    if (pGui->addDropdown("##Resolution", mResolutions, *reinterpret_cast<uint32*> (&CurrentResolution), true))
    {
        const ivec2 NewResolution = kResolutions[CurrentResolution];
        pSample->getWindow()->resize(NewResolution.x, NewResolution.y);
    }

    pGui->addSeparator();

    pGui->addCheckBox("Use Rdae (only with 720p)", mUseRdae);
    pGui->addTooltip("If unchecked SVGF is used", true);

    if (pGui->addCheckBox("Use TAA", mUseTAA))
        mPass.pGBuffer->setPatternGenerator(mUseTAA ? HaltonSamplePattern::create() : nullptr);
    pGui->addTooltip("Enable/Disable Temporal Anti-Aliasing", true);

    pGui->addSeparator();

    // Pass Guis
    for (auto& pPass : mPasses)
    {
        ImGui::PushID(pPass);
        if (pGui->beginGroup(pPass->getDesc()))
        {
            pPass->onGuiRender(pGui);
            pGui->endGroup();
        }
        ImGui::PopID();
    }

    mTAA.pTAA->renderUI(pGui, "TAA");
}

bool SSTDemo::loadScene(RenderContext* pRenderContext, const std::string& path)
{
    RtScene::SharedPtr pScene = RtScene::loadFromFile(path, RtBuildFlags::None, Model::LoadFlags::None, Scene::LoadFlags::GenerateAreaLights);

    if (!pScene)
        return false;

    mpScene = pScene;

    // Adjust aspect ratio
    mpScene->setCamerasAspectRatio(static_cast<float> (mPassData.getWidth()) / static_cast<float> (mPassData.getHeight()));

    // Update camera
    mpCamera = pScene->getActiveCamera();
    if (mpCamera)
    {
        mCameraController.attachCamera(mpCamera);
        mpCameraPath = nullptr;
        // Detach camera from path to enable free view
        for (uint32_t i = 0; i < pScene->getPathCount(); i++)
            if (mpCamera->getAttachedPath())
                if (pScene->getPath(i)->getName() == mpCamera->getAttachedPath()->getName())
                    mpCameraPath = pScene->getPath(i);
        mpCameraPath->detachObject(mpCamera);
    }

    // Set scene sampler
    Sampler::Desc desc;
    desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    desc.setLodParams(0, D3D12_FLOAT32_MAX, 0);
    desc.setMaxAnisotropy(16);
    desc.setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
    desc.setComparisonMode(Sampler::ComparisonMode::LessEqual);
    mpScene->bindSampler(Sampler::create(desc));

    for (auto& pPass : mPasses)
        pPass->setScene(pRenderContext, mpScene);

    mPass.pGBuffer->setPatternGenerator(mUseTAA ? HaltonSamplePattern::create() : nullptr);

    return true;
}

void SSTDemo::onFrameRender(SampleCallbacks* pCallbacks, RenderContext* pRenderContext, const std::shared_ptr<Fbo>& pTargetFbo)
{
    if (mpScene) mpScene->update(pCallbacks->getCurrentTime(), &mCameraController);

    auto pColor = asTexture(mPassData["gColor"]);
    auto pPrevColor = asTexture(mPassData["gPrevColor"]);
    pRenderContext->clearRtv(pColor->getRTV().get(), vec4(0.f, 0.f, 0.f, 1.f));

    // Execute passes
    mPass.pGBuffer->onFrameRender(pRenderContext, mPassData);
    mPass.pVPLTracing->onFrameRender(pRenderContext, mPassData);
    mPass.pVPLTree->onFrameRender(pRenderContext, mPassData);
    mPass.pVPLSampling->onFrameRender(pRenderContext, mPassData);

    if (mUseRdae)
    {
        mPass.pRdae->onFrameRender(pRenderContext, mPassData);
        mPass.pTemporalFilter->onFrameRender(pRenderContext, mPassData);
        pRenderContext->blit(asTexture(mPassData["gTemporalFiltered"])->getSRV(), pColor->getRTV());
    }
    else
    {
        mPass.pSVGF->onFrameRender(pRenderContext, mPassData);
        pRenderContext->blit(asTexture(mPassData["gFilteredSVGF"])->getSRV(), pColor->getRTV());
    }


    if (mUseTAA)
    {
        PROFILE("TAA");
        //  Get the Current Color and Motion Vectors
        const Texture::SharedPtr pCurColor  = asTexture(mPassData["gColor"]);
        const Texture::SharedPtr pMotionVec = asTexture(mPassData["gMotion"]);

        //  Get the Previous Color
        const Texture::SharedPtr pPrevColor = mTAA.getInactiveFbo()->getColorTexture(0);

        //  Execute the Temporal Anti-Aliasing
        pRenderContext->getGraphicsState()->pushFbo(mTAA.getActiveFbo());
        mTAA.pTAA->execute(pRenderContext, pCurColor, pPrevColor, pMotionVec);
        pRenderContext->getGraphicsState()->popFbo();

        //  Copy over the Anti-Aliased Color Texture
        pRenderContext->blit(mTAA.getActiveFbo()->getColorTexture(0)->getSRV(0, 1), pCurColor->getRTV());

        //  Swap the Fbos
        mTAA.switchFbos();
    }

    mPass.pVPLVisualizer->onFrameRender(pRenderContext, mPassData);

    // Blit output texture to ouptut framebuffer
    auto pOutput = asTexture(mPassData[mCurrentOutput]);

    if (pOutput)
    {
        auto bindFlags = pOutput->getBindFlags();
        if (pOutput->getType() == Resource::Type::Texture2D && (is_set(bindFlags, ResourceBindFlags::ShaderResource) || is_set(bindFlags, ResourceBindFlags::UnorderedAccess)))
            pRenderContext->blit(pOutput->getSRV(), pTargetFbo->getRenderTargetView(0));
    }
}

bool SSTDemo::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    bool handled = false;

    handled |= mCameraController.onMouseEvent(mouseEvent);

    for (auto& pPass : mPasses)
        handled |= pPass->onMouseEvent(mouseEvent);

    switch (mouseEvent.type)
    {
    case MouseEvent::Type::Wheel: // Zoom
        mpCamera->setFocalLength(mpCamera->getFocalLength() + 5.f * mouseEvent.wheelDelta.y * (mpCamera->getFocalLength() * 0.01f));
        handled = true;
        break;
    default:
        break;
    }
    return handled;
}

bool SSTDemo::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    bool handled = false;
    handled |= mCameraController.onKeyEvent(keyEvent);

    for (auto& pPass : mPasses)
        handled |= pPass->onKeyEvent(keyEvent);

    return handled;
}

void SSTDemo::onDataReload(SampleCallbacks* pSample)
{
    for (auto& pPass : mPasses)
        pPass->onDataReload();
}

void SSTDemo::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    mPassData.setWidth(width);
    mPassData.setHeight(height);

    createResources();

    for (auto& pPass : mPasses)
        pPass->onResizeSwapChain(width, height, mPassData);

    mTAA.pTAA->onResize(width, height);

    // Release the TAA FBOs
    mTAA.resetFbos();

    Fbo::Desc taaFboDesc;
    taaFboDesc.setColorTarget(0, ResourceFormat::RGBA32Float);
    mTAA.createFbos(width, height, taaFboDesc);

    mPass.pGBuffer->setPatternGenerator(mUseTAA ? HaltonSamplePattern::create() : nullptr);
}

void SSTDemo::createResources()
{
    const auto width = mPassData.getWidth();
    const auto height = mPassData.getHeight();

    const auto bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget;
    auto pColor = Texture::create2D(width, height, ResourceFormat::RGBA32Float, 1u, 1u, nullptr, bindFlags);

    mPassData.addResource("gColor", pColor);
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    //AllocConsole();
    //AttachConsole(GetCurrentProcessId());
    //freopen("CON", "w", stdout);

    SSTDemo::UniquePtr pSSTDemo = std::make_unique<SSTDemo>();
    SampleConfig config;

    config.windowDesc.title = "SSTDemo";
    config.windowDesc.resizableWindow = false;
    config.windowDesc.height = 720;
    config.windowDesc.width = 1280;

    Sample::run(config, pSSTDemo);
    return 0;
}
