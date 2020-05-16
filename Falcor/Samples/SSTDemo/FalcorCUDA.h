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

#include "Falcor.h"

#define checkCudaError(x) Falcor::Cuda::checkError(x, __FILE__, __LINE__)

namespace FalcorCUDA
{
    #include "cuda_runtime_api.h"

    bool mapResource(Falcor::Resource* resource, unsigned long long sizeInBytes, cudaExternalMemory_t& externalMemory, void** pCudaDev);
    bool unmapResource(cudaExternalMemory_t& externalMemory);

    void importSemaphore(Falcor::GpuFence* pFence, cudaExternalSemaphore_t& externalSemaphore);
    void destroyExternalSemaphore(cudaExternalSemaphore_t& externalSemaphore);
};

namespace Falcor
{
    struct Cuda
    {
        static void checkError(FalcorCUDA::cudaError_t error, const char* file, const int line);

        static void* malloc(size_t size);
        static void free(void* buffer);
        static void memset(void* buffer, int value, size_t count);
        static void memcpy(void* dst, const void* src, size_t count, enum FalcorCUDA::cudaMemcpyKind kind);
    };
}
