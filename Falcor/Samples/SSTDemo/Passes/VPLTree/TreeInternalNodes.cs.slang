
/***************************************************************************
# The MIT License(MIT)
#
# Copyright(c) 2019 Toru Niina
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
***************************************************************************/

#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"
#include "../Shared/VPLData.h"
#include "../Shared/VPLTreeStructs.h"


const StructuredBuffer<uint64_t> gCodes;
const StructuredBuffer<VPLStats> gVPLStats;
RWStructuredBuffer<TreeNode> gNodes;


int clz(uint x)
{
   return 31 - firstbithigh(x);
}

int clz64(uint64_t x)
{
    const int h = clz((uint) (x >> 32));
    const int l = clz((uint) (x & 0xFFFFFFFF));
    return h == 32 ? h + l : h;
}

void swap(inout uint u1, inout uint u2)
{
    uint tmp = u1;
    u1 = u2;
    u2 = tmp;
}

int common_upper_bits(const uint64_t lhs, const uint64_t rhs)
{
    return clz64(lhs ^ rhs);
}

uint2 determine_range(const uint num_leaves, const uint max_leaves, uint idx)
{
    if (idx == 0)
        return uint2(0, num_leaves - 1);

    // determine direction of the range
    const uint64_t self_code = gCodes[idx];
    const int L_delta = common_upper_bits(self_code, gCodes[idx - 1]);
    const int R_delta = common_upper_bits(self_code, gCodes[idx + 1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if (0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits(self_code, gCodes[i_tmp]);
    }
    while (delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, gCodes[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while (t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, gCodes[i_tmp]);
        }
        if (delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    uint jdx = idx + l * d;
    if (d < 0)
    {
        swap(idx, jdx); // make it sure that idx < jdx
    }
    return uint2(idx, jdx);
}

uint find_split(const uint first, const uint last)
{
    const uint64_t first_code = gCodes[first];
    const uint64_t last_code  = gCodes[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = common_upper_bits(first_code, gCodes[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

[numthreads(256, 1, 1)]
void treeInternalNodes(uint3 DTid : SV_DispatchThreadID)
{
    const int idx = DTid.x;
    const int num_objects = gVPLStats[0].numVPLs;

    if (idx >= max(0, num_objects - 1))
        return;

    gNodes[idx].vpl_idx = MAX_VPLS + idx; //  assign internal node storage
    const uint2 ij = determine_range(num_objects, MAX_VPLS, idx);
    const int gamma = find_split(ij.x, ij.y);

    gNodes[idx].left_idx = gamma;
    gNodes[idx].right_idx = gamma + 1;

    if (min(ij.x, ij.y) == gamma)
        gNodes[idx].left_idx += MAX_VPLS - 1;
    if (max(ij.x, ij.y) == gamma + 1)
        gNodes[idx].right_idx += MAX_VPLS - 1;

    gNodes[gNodes[idx].left_idx].parent_idx = idx;
    gNodes[gNodes[idx].right_idx].parent_idx = idx;
}
