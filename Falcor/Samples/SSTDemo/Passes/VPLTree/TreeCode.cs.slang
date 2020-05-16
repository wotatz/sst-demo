#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"
#include "../Shared/VPLData.h"
#include "Codes.slangh"

const StructuredBuffer<VPLData>  gVPLData;
RWStructuredBuffer<uint64_t>     gCodes;

cbuffer CB
{
    float3 gMinExtent;
    float3 gMaxExtent;
}

[numthreads(256, 1, 1)]
void treeCode(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= MAX_VPLS) return;

    const int id = gVPLData[DTid.x].id;
    if (id < 0) // VPL is not valid
    {
        // Set code to maximum so it will always end up in the end after sorting.
        gCodes[DTid.x] = uint64_t(-1);
        return;
    }

    // Transform position
    float3 lower = gMinExtent;
    float3 upper = gMaxExtent;

    float3 position = gVPLData[DTid.x].getPosW();
    position -= lower;
    position /= upper - lower;

    // Compute morton code
    const uint64_t mortonCode = morton_code(position);

    // Compute direction code
    const float3 normal = gVPLData[DTid.x].getNormW();
    const uint64_t dirCode = direction_code(normal, NUM_SPHERE_SECTIONS);
 
    // Compute id code
    const uint64_t idCode = id;
   
    // Combine codes and store
    gCodes[DTid.x] = (mortonCode << BEGIN_MORTON_BITS) | (dirCode << BEGIN_DIR_BITS) | (idCode << BEGIN_ID_BITS);
}
