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

cbuffer CBCommon
{
    uint NumElements;
}

// Takes Value and widens it by one bit at the location of the bit
// in the mask.  A one is inserted in the space.  OneBitMask must
// have one and only one bit set.
uint InsertOneBit(uint Value, uint OneBitMask)
{
    uint Mask = OneBitMask - 1;
    return (Value & ~Mask) << 1 | (Value & Mask) | OneBitMask;
}

bool ShouldSwap(uint64_t A, uint64_t B)
{
    return (A & COMP_MASK) > (B & COMP_MASK);
}
