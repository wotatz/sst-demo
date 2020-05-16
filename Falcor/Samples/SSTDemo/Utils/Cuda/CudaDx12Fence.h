#include "Falcor.h"
#include "FalcorCUDA.h"
#include <cstddef>
#include <memory>
#include <utility>

class CudaDx12Fence
{
public:
    using SharedPtr = std::shared_ptr<CudaDx12Fence>;

    static CudaDx12Fence::SharedPtr create();
    ~CudaDx12Fence();

    uint64_t signalCommandQueue(Falcor::CommandQueueHandle& commandQueue);
    uint64_t waitCommandQueue(Falcor::CommandQueueHandle& commandQueue);

    uint64_t signalStream(FalcorCUDA::cudaStream_t& stream);
    uint64_t waitStream(FalcorCUDA::cudaStream_t& stream);

    Falcor::GpuFence::SharedPtr getGpuFence() { return mpFence; }

    CudaDx12Fence(const CudaDx12Fence&) = delete;
    CudaDx12Fence& operator=(const CudaDx12Fence&) = delete;

private:
    CudaDx12Fence();
    Falcor::GpuFence::SharedPtr mpFence;
    FalcorCUDA::cudaExternalSemaphore_t mExternalSemaphore = nullptr;
};
