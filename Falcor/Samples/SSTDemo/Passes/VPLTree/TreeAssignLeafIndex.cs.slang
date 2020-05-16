#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"
#include "../Shared/VPLData.h"
#include "../Shared/VPLTreeStructs.h"

const StructuredBuffer<uint64_t> gCodes;
RWStructuredBuffer<TreeNode>     gNodes;


[numthreads(256, 1, 1)]
void treeAssignLeafIndex(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= MAX_VPLS) return;

    const uint64_t m64 = gCodes[DTid.x];
    if (m64 == uint64_t(-1))  // Invalid VPL!
        return;

    const uint idx = gCodes[DTid.x] & getBitMask(NUM_ID_BITS);

    // Assign vpl index to leaf node (node buffer : [internal nodes, leaf nodes])
    const int numInternalNodes = getNumInternalNodes(MAX_VPLS);
    gNodes[DTid.x + numInternalNodes].vpl_idx = idx;
}
