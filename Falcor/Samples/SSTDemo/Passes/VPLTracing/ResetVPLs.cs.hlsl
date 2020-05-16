#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"

#include "Passes/Shared/VPLData.h"


RWStructuredBuffer<VPLData>  gVPLData;
RWStructuredBuffer<float3>   gVPLPositions;
RWStructuredBuffer<VPLStats> gVPLStats;

cbuffer CB
{
    uint gMaxVPLs;
}

[numthreads(256, 1, 1)]
void resetVPLs(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gMaxVPLs) return;
    if (DTid.x == 0) gVPLStats[0].numVPLs = 0;

    gVPLData[DTid.x].id = -1;
    gVPLPositions[DTid.x] = float3(FLT_MAX);
}
