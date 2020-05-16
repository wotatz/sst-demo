/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "FalcorCUDA.h"
#include <AccCtrl.h>
#include <aclapi.h>

using namespace Falcor;

#define CU_CHECK_SUCCESS(x)                                               \
    do {                                                                  \
        FalcorCUDA::CUresult result = x;                                  \
        if (result != FalcorCUDA::CUDA_SUCCESS)                           \
        {                                                                 \
            const char *msg;                                              \
            FalcorCUDA::cuGetErrorName(result, &msg);                     \
            logError("CUDA Error: failed with error " + std::string(msg));\
        }                                                                 \
    } while(0)

#define CUDA_CHECK_SUCCESS(x)                                                     \
    do {                                                                          \
        FalcorCUDA::cudaError_t result = x;                                       \
        if (result != FalcorCUDA::cudaSuccess)                                    \
        {                                                                         \
            const char *msg = FalcorCUDA::cudaGetErrorString(result);             \
            Falcor::logError("CUDA Error: failed with error " + std::string(msg));\
        }                                                                         \
    } while(0)

namespace FalcorCUDA
{
    //#include <cuda.h>
    bool mapResource(Falcor::Resource* resource, unsigned long long sizeInBytes, cudaExternalMemory_t& externalMemory, void** pCudaDev)
    {
        using namespace Falcor;
        ID3D12Device* pDevice = Falcor::gpDevice->getApiHandle();

        ID3D12Resource* d3d12resource = resource->getApiHandle().GetInterfacePtr();
        D3D12_RESOURCE_DESC desc = resource->getApiHandle()->GetDesc();
        unsigned long long bufferSize = sizeInBytes;
        HANDLE sharedHandle;
        LPCWSTR name = NULL;

        SECURITY_ATTRIBUTES securityAttributes;
        securityAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
        securityAttributes.lpSecurityDescriptor = NULL;
        securityAttributes.bInheritHandle = false;

        d3d_call(pDevice->CreateSharedHandle(d3d12resource, &securityAttributes, GENERIC_ALL, name, &sharedHandle));

        D3D12_RESOURCE_ALLOCATION_INFO d3d12ResourceAllocationInfo;
        d3d12ResourceAllocationInfo = pDevice->GetResourceAllocationInfo(0, 1, &desc);
        size_t actualSize = d3d12ResourceAllocationInfo.SizeInBytes;
        size_t alignment = d3d12ResourceAllocationInfo.Alignment;

        cudaExternalMemoryHandleDesc externalMemoryHandleDesc;
        memset(&externalMemoryHandleDesc, 0, sizeof(externalMemoryHandleDesc));
        externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
        externalMemoryHandleDesc.handle.win32.handle = sharedHandle;
        externalMemoryHandleDesc.size = actualSize;
        externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;

        CUDA_CHECK_SUCCESS(cudaImportExternalMemory(&externalMemory, &externalMemoryHandleDesc));

        cudaExternalMemoryBufferDesc externalMemoryBufferDesc;
        memset(&externalMemoryBufferDesc, 0, sizeof(externalMemoryBufferDesc));
        externalMemoryBufferDesc.offset = 0;
        externalMemoryBufferDesc.size = bufferSize;
        externalMemoryBufferDesc.flags = 0;

        CUDA_CHECK_SUCCESS(cudaExternalMemoryGetMappedBuffer(pCudaDev, externalMemory, &externalMemoryBufferDesc));
        return true;
    }

    bool unmapResource(cudaExternalMemory_t& externalMemory)
    {
        if (externalMemory)
        {
            CUDA_CHECK_SUCCESS(cudaDestroyExternalMemory(externalMemory));
            externalMemory = nullptr;
            return true;
        }
        return false;
    }

    void importSemaphore(Falcor::GpuFence* pFence, cudaExternalSemaphore_t& externalSemaphore)
    {
        using namespace Falcor;
        SECURITY_ATTRIBUTES securityAttributes;
        securityAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
        securityAttributes.lpSecurityDescriptor = NULL;
        securityAttributes.bInheritHandle = false;

        ID3D12Device* pDevice = Falcor::gpDevice->getApiHandle();
        ID3D12Fence* pD3D12Fence = pFence->getApiHandle().GetInterfacePtr();

        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
        std::memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));

        LPCWSTR name = NULL;
        HANDLE sharedHandle;
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;

        d3d_call(pDevice->CreateSharedHandle(pD3D12Fence, &securityAttributes, GENERIC_ALL, name, &sharedHandle));
        externalSemaphoreHandleDesc.handle.win32.handle = (void*)sharedHandle;
        externalSemaphoreHandleDesc.flags = 0;

        CUDA_CHECK_SUCCESS(cudaImportExternalSemaphore(&externalSemaphore, &externalSemaphoreHandleDesc));
    }

    void destroyExternalSemaphore(cudaExternalSemaphore_t& externalSemaphore)
    {
        if (externalSemaphore) CUDA_CHECK_SUCCESS(cudaDestroyExternalSemaphore(externalSemaphore));
    }
}

namespace Falcor
{
    using namespace FalcorCUDA;

    void Cuda::checkError(FalcorCUDA::cudaError_t error, const char* file, const int line)
    {
        if (error != cudaSuccess)
        {
            std::cout << "[" << file << ":" << line << "] got CUDA error " << error << ": " << cudaGetErrorString(error)
                << std::endl;
            assert(false);
        }
    }

    void* Cuda::malloc(size_t size)
    {
        void* ptr = nullptr;
        CUDA_CHECK_SUCCESS(cudaMalloc(&ptr, size));
        return ptr;
    }

    void Cuda::free(void* buffer)
    {
        CUDA_CHECK_SUCCESS(cudaFree(buffer));
    }

    void Cuda::memset(void* buffer, int value, size_t count)
    {
        CUDA_CHECK_SUCCESS(cudaMemset(buffer, value, count));
    }

    void Cuda::memcpy(void* dst, const void* src, size_t count, enum FalcorCUDA::cudaMemcpyKind kind)
    {
        CUDA_CHECK_SUCCESS(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    }
}
