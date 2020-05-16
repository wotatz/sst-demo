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

#include "BitonicSort.h"
#include <sstream>

namespace
{
    unsigned long upper_power_of_two(unsigned long v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

    unsigned int log2(unsigned int v)
    {
        int targetlevel = 0;
        while (v >>= 1) ++targetlevel;
        return targetlevel;
    }
}


static const char kInnerShaderFilename[]        = "Passes/VPLTree/Sort/BitonicInnerSort.cs.slang";
static const char kOuterShaderFilename[]        = "Passes/VPLTree/Sort/BitonicOuterSort.cs.slang";
static const char kPreSortShaderFilename[]      = "Passes/VPLTree/Sort/BitonicPreSort.cs.slang";
static const char kIndirectArgsShaderFilename[] = "Passes/VPLTree/Sort/BitonicIndirectArgs.cs.slang";

BitonicSort::BitonicSort()
{
    mSort.pState = ComputeState::create();

    // Create shaders
    mProgramDefineList.add("COMP_MASK",  "0xFFFFFFFFFFFFFFFF");
    mProgramDefineList.add("NULL_ITEM" , "0xFFFFFFFFFFFFFFFF");

    const std::string SM = "6_0";
    mSort.pInnerProgram        = ComputeProgram::createFromFile(kInnerShaderFilename,        "main", mProgramDefineList, Shader::CompilerFlags::None, SM);
    mSort.pOuterProgram        = ComputeProgram::createFromFile(kOuterShaderFilename,        "main", mProgramDefineList, Shader::CompilerFlags::None, SM);
    mSort.pPreSortProgram      = ComputeProgram::createFromFile(kPreSortShaderFilename,      "main", mProgramDefineList, Shader::CompilerFlags::None, SM);
    mSort.pIndirectArgsProgram = ComputeProgram::createFromFile(kIndirectArgsShaderFilename, "main", mProgramDefineList, Shader::CompilerFlags::None, SM);

    mSort.pInnerVars        = ComputeVars::create(mSort.pInnerProgram->getReflector());
    mSort.pOuterVars        = ComputeVars::create(mSort.pOuterProgram->getReflector());
    mSort.pPreSortVars      = ComputeVars::create(mSort.pPreSortProgram->getReflector());
    mSort.pIndirectArgsVars = ComputeVars::create(mSort.pIndirectArgsProgram->getReflector());

    mpBufferIndirectArgs = Buffer::create(12 * 22 * 23 * 100 / 2, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::IndirectArg | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None);
}

BitonicSort::SharedPtr BitonicSort::create()
{
    return SharedPtr(new BitonicSort());
}

bool BitonicSort::execute(RenderContext* pRenderContext, StructuredBuffer::SharedPtr pData, uint32_t totalSize, int2 bitRange, uint32_t chunkSize, uint32_t groupSize)
{
    PROFILE("BitonicSort");

    const uint32_t MaxNumElements = totalSize;
    const uint32_t AlignedMaxNumElements = upper_power_of_two(MaxNumElements);
    const uint32_t MaxIterations = log2(std::max(2048u, AlignedMaxNumElements)) - 10;

    // Setup compare bit mask
    const uint numBits = bitRange.x - bitRange.y + 1;
    uint64_t mask = numBits == 64 ? UINT64_MAX : (1ull << numBits) - 1;
    mask <<= bitRange.y;

    std::stringstream ss;
    ss << "0x" << std::hex << mask;

    // Set program defines
    mProgramDefineList.add("COMP_MASK", ss.str());
    mProgramDefineList.add("NULL_ITEM", "0xFFFFFFFFFFFFFFFF");

    mSort.pIndirectArgsProgram->addDefines(mProgramDefineList);
    mSort.pPreSortProgram->addDefines(mProgramDefineList);
    mSort.pInnerProgram->addDefines(mProgramDefineList);
    mSort.pOuterProgram->addDefines(mProgramDefineList);

    // Generate execute indirect arguments
    mSort.pState->setProgram(mSort.pIndirectArgsProgram);
    mSort.pIndirectArgsVars->setRawBuffer("g_IndirectArgsBuffer", mpBufferIndirectArgs);
    mSort.pIndirectArgsVars["CB"]["MaxIterations"]     = MaxIterations;
    mSort.pIndirectArgsVars["CBCommon"]["NumElements"] = MaxNumElements;

    pRenderContext->setComputeState(mSort.pState);
    pRenderContext->setComputeVars(mSort.pIndirectArgsVars);
    pRenderContext->dispatch(1, 1, 1);

    // Pre-Sort the buffer up to k = 2048. 
    mSort.pState->setProgram(mSort.pPreSortProgram);
    mSort.pPreSortVars->setStructuredBuffer("g_SortBuffer", pData);
    mSort.pPreSortVars["CBCommon"]["NumElements"] = MaxNumElements;

    pRenderContext->setComputeState(mSort.pState);
    pRenderContext->setComputeVars(mSort.pPreSortVars);
    pRenderContext->dispatchIndirect(mpBufferIndirectArgs.get(), 0);

    pRenderContext->uavBarrier(pData.get());

    uint32_t IndirectArgsOffset = 12;

    for (uint32_t k = 4096; k <= AlignedMaxNumElements; k *= 2)
    {
        mSort.pState->setProgram(mSort.pOuterProgram);

        for (uint32_t j = k / 2; j >= 2048; j /= 2)
        {
            mSort.pOuterVars->setStructuredBuffer("g_SortBuffer", pData);
            mSort.pOuterVars["CB"]["k"] = k;
            mSort.pOuterVars["CB"]["j"] = j;
            mSort.pOuterVars["CBCommon"]["NumElements"] = MaxNumElements;

            pRenderContext->setComputeState(mSort.pState);
            pRenderContext->setComputeVars(mSort.pOuterVars);
            pRenderContext->uavBarrier(pData.get());
            pRenderContext->dispatchIndirect(mpBufferIndirectArgs.get(), IndirectArgsOffset);

            IndirectArgsOffset += 12;
        }

        mSort.pState->setProgram(mSort.pInnerProgram);

        mSort.pInnerVars->setStructuredBuffer("g_SortBuffer", pData);
        mSort.pInnerVars["CB"]["k"] = k;
        mSort.pInnerVars["CBCommon"]["NumElements"] = MaxNumElements;

        pRenderContext->setComputeState(mSort.pState);
        pRenderContext->setComputeVars(mSort.pInnerVars);
        pRenderContext->uavBarrier(pData.get());
        pRenderContext->dispatchIndirect(mpBufferIndirectArgs.get(), IndirectArgsOffset);

        IndirectArgsOffset += 12;
    }
    return true;
}
