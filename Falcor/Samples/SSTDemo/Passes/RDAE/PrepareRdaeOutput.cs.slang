#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"

const StructuredBuffer<float4> gRdaeOutput;
RWTexture2D<float4>            gOutput;

cbuffer PerFrameCB
{
  float gExponent;
  int2  gWindowDims;
  int2  gCNNDims;
};

[numthreads(32, 32, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadId)
{
    uint2 posStart = groupId.xy * 32;
    uint2 pixelPos = posStart + groupThreadId.xy;

    if (pixelPos.x >= gWindowDims.x || pixelPos.y >= gWindowDims.y)
        return;

    const int bufferIndex = pixelPos.y * gCNNDims.x + pixelPos.x;
    gOutput[pixelPos]  = float4(pow(gRdaeOutput[bufferIndex].rgb, gExponent), 1.f);
}
