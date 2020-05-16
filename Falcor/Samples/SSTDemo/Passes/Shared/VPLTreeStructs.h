#pragma once

#include "../Shared/VPLData.h"

#if defined(HOST_CODE)
#include "Falcor.h"
#endif

struct TreeNode
{
    uint parent_idx; // parent node
    uint left_idx;   // index of left  child node
    uint right_idx;  // index of right child node
    uint vpl_idx;    // == 0xFFFFFFFF if internal node.
    uint flag;       // got node already processed?
};

struct VPLMerge
{
    float2 ApproxScore;
};

struct TreeApproxParams
{
    float minNormalScore  DEFAULTS(0.25f);
    float maxNormalZStd   DEFAULTS(0.1f);
    float pad2;
    float pad3;
};

inline uint numDirCodeBits(const int sections)
{
    return (3 + 2 * sections + 1) + 1;
}

inline uint getNumInternalNodes(uint maxVPLs)
{
    return maxVPLs - 1;
}

inline uint getNumTotalNodes(uint maxVPLs)
{
    return maxVPLs * 2 - 1;
}

inline uint64_t getBitMask(uint numBitsSet)
{
    return (1ull << numBitsSet) - 1;
}
