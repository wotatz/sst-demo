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
#include "FalcorExperimental.h"

using namespace Falcor;

class BitonicSort : public std::enable_shared_from_this<BitonicSort>
{
public:
    using SharedPtr = std::shared_ptr<BitonicSort>;
    using SharedConstPtr = std::shared_ptr<const BitonicSort>;
    virtual ~BitonicSort() = default;

    /** Create a new bitonic sort object.
        \return New object, or throws an exception on error.
    */
    static SharedPtr create();

    /** In-place bitonic sort in chunks of N elements. Each chunk is sorted in ascending order.
        \param[in] pRenderContext The render context.
        \param[in] pData The data buffer to sort in-place.
        \param[in] totalSize The total number of elements in the buffer. This does _not_ have to be a multiple of chunkSize.
        \param[in] bitRange  The most and least-significant bit index for comparision.
        \param[in] chunkSize The number of elements per chunk. Each chunk is individually sorted. Must be a power-of-two in the range [1, groupSize].
        \param[in] groupSize Thread group size. Must be a power-of-two in the range [1,1024]. The default group size of 256 is generally the fastest.
        \return True if successful, false if an error occured.
    */
    bool execute(RenderContext* pRenderContext, StructuredBuffer::SharedPtr pData, uint32_t totalSize, int2 bitRange, uint32_t chunkSize, uint32_t groupSize = 256);

protected:
    BitonicSort();

    struct
    {
        ComputeProgram::SharedPtr pIndirectArgsProgram;
        ComputeProgram::SharedPtr pInnerProgram;
        ComputeProgram::SharedPtr pOuterProgram;
        ComputeProgram::SharedPtr pPreSortProgram;

        ComputeVars::SharedPtr pIndirectArgsVars;
        ComputeVars::SharedPtr pInnerVars;
        ComputeVars::SharedPtr pOuterVars;
        ComputeVars::SharedPtr pPreSortVars;

        ComputeState::SharedPtr pState;
    } mSort;

    Program::DefineList mProgramDefineList;

    Buffer::SharedPtr mpBufferIndirectArgs;
};
