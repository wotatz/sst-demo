#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"
#include "../Shared/VPLData.h"
#include "../Shared/VPLTreeStructs.h"

RWStructuredBuffer<TreeNode>  gNodes;
RWStructuredBuffer<VPLMerge>  gMerge;

[numthreads(256, 1, 1)]
void treeInit(uint3 DTid : SV_DispatchThreadID)
{
    const int maxVPLs          = MAX_VPLS;
    const int numInternalNodes = getNumInternalNodes(maxVPLs);
    const int numTotalNodes    = getNumTotalNodes(maxVPLs);

    if (DTid.x >= numTotalNodes)
        return;

    // Init nodes
    TreeNode node;
    node.parent_idx = 0xFFFFFFFF;
    node.left_idx   = 0xFFFFFFFF;
    node.right_idx  = 0xFFFFFFFF;
    node.vpl_idx    = 0xFFFFFFFF;
    node.flag       = 0x0;
    gNodes[DTid.x]  = node;

    // Init merge data
    VPLMerge merge;
    merge.ApproxScore = float2(1.f, 0.f); // Normal score / Normal Z std
    gMerge[DTid.x] = merge;
}
