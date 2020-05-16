#include "CudaDx12Fence.h"

using namespace FalcorCUDA;

CudaDx12Fence::SharedPtr CudaDx12Fence::create()
{
    CudaDx12Fence::SharedPtr pSharedFence = SharedPtr(new CudaDx12Fence);
    return pSharedFence;
}

CudaDx12Fence::CudaDx12Fence()
{
    mpFence = Falcor::GpuFence::create(/*Shared*/ true);
    FalcorCUDA::importSemaphore(mpFence.get(), mExternalSemaphore);
}

CudaDx12Fence::~CudaDx12Fence()
{
    FalcorCUDA::destroyExternalSemaphore(mExternalSemaphore);
}

uint64_t CudaDx12Fence::signalCommandQueue(Falcor::CommandQueueHandle& commandQueue)
{
    return mpFence->gpuSignal(commandQueue);
}

uint64_t CudaDx12Fence::waitCommandQueue(Falcor::CommandQueueHandle& commandQueue)
{
    mpFence->syncGpu(commandQueue);
    return mpFence->getCpuValue() - 1;
}

uint64_t CudaDx12Fence::signalStream(FalcorCUDA::cudaStream_t& stream)
{
    FalcorCUDA::cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));

    const uint64_t signalValue = mpFence->gpuExternalSignaled();

    signalParams.params.fence.value = signalValue;
    checkCudaError(cudaSignalExternalSemaphoresAsync(&mExternalSemaphore, &signalParams, 1, stream));
    return signalValue;
}

uint64_t CudaDx12Fence::waitStream(FalcorCUDA::cudaStream_t& stream)
{
    FalcorCUDA::cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));

    const auto waitValue = mpFence->getCpuValue() - 1;

    waitParams.params.fence.value = waitValue;
    checkCudaError(cudaWaitExternalSemaphoresAsync(&mExternalSemaphore, &waitParams, 1, stream));
    return waitValue;
}
