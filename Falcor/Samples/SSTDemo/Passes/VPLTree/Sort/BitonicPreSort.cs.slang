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

groupshared uint64_t gs_SortValues[2048];


void FillSortKey(uint Element, uint ListCount)
{
    // Unused elements must sort to the end
    if (Element < ListCount)
    {
        uint64_t value = g_SortBuffer[Element];
        gs_SortValues[Element & 2047] = value;
    }
    else
    {
        gs_SortValues[Element & 2047] = NULL_ITEM;
    }
}

void StoreKeyIndexPair(uint Element, uint ListCount)
{
    if (Element < ListCount)
        g_SortBuffer[Element] = gs_SortValues[Element & 2047];
}


[numthreads(1024, 1, 1)]
void main(uint3 Gid : SV_GroupID, uint GI : SV_GroupIndex)
{
    // Item index of the start of this group
    const uint GroupStart = Gid.x * 2048;

    // Actual number of items that need sorting
    const uint ListCount = NumElements;

    FillSortKey(GroupStart + GI, ListCount);
    FillSortKey(GroupStart + GI + 1024, ListCount);

    GroupMemoryBarrierWithGroupSync();

    uint k;

    // This is better unrolled because it reduces ALU and because some
    // architectures can load/store two LDS items in a single instruction
    // as long as their separation is a compile-time constant.
    [unroll]
    for (k = 2; k <= 2048; k <<= 1)
    {
        [unroll]
        for (uint j = k / 2; j > 0; j /= 2)
        {
            uint Index2 = InsertOneBit(GI, j);
            uint Index1 = Index2 ^ (k == 2 * j ? k - 1 : j);

            uint64_t A = gs_SortValues[Index1];
            uint64_t B = gs_SortValues[Index2];

            if (ShouldSwap(A, B))
            {
                // Swap the keys
                gs_SortValues[Index1] = B;
                gs_SortValues[Index2] = A;
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Write sorted results to memory
    StoreKeyIndexPair(GroupStart + GI, ListCount);
    StoreKeyIndexPair(GroupStart + GI + 1024, ListCount);
}
