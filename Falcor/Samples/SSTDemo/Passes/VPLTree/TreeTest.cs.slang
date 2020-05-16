#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"
#include "../Shared/VPLData.h"
#include "../Shared/VPLTreeStructs.h"

const StructuredBuffer<VPLStats>   gVPLStats;
const StructuredBuffer<VPLData>    gVPLDataPacked;
//RWStructuredBuffer<VPLDataFull>    gVPLDataFull;


[numthreads(256, 1, 1)]
void treeTest(uint3 DTid : SV_DispatchThreadID)
{
    const int id = DTid.x;
    if (id >= (MAX_VPLS * 2 -1)) return;

    //gVPLDataFull[id].posW = gVPLDataPacked[id].getPosW();
    //gVPLDataFull[id].normW = gVPLDataPacked[id].getNormW();
    //gVPLDataFull[id].color = gVPLDataPacked[id].getColor();
    //gVPLDataFull[id].intensity = gVPLDataPacked[id].getIntensity();
    //gVPLDataFull[id].aabbPosWMin = gVPLDataPacked[id].getAABBMin();
    //gVPLDataFull[id].aabbPosWMax = gVPLDataPacked[id].getAABBMax();
    //gVPLDataFull[id].variance = gVPLDataPacked[id].getVariance();
    //gVPLDataFull[id].earlyStop = gVPLDataPacked[id].getEarlyStop();
}

