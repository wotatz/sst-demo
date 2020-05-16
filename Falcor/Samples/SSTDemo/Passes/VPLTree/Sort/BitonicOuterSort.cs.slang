//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
// Developed by Minigraph
//
// Author(s):  James Stanard

#include "BitonicCommon.slang"

RWStructuredBuffer<uint64_t> g_SortBuffer;

cbuffer CB
{
    uint k;    // k >= 4096
    uint j;    // j >= 2048 && j < k
};


[numthreads(1024, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    const uint ListCount = NumElements;

    // Form unique index pair from dispatch thread ID
    uint Index2 = InsertOneBit(DTid.x, j);
    uint Index1 = Index2 ^ (k == 2 * j ? k - 1 : j);

    if (Index2 >= ListCount)
        return;

    uint64_t A = g_SortBuffer[Index1];
    uint64_t B = g_SortBuffer[Index2];

    if (ShouldSwap(A, B))
    {
        g_SortBuffer[Index1] = B;
        g_SortBuffer[Index2] = A;
    }
}
