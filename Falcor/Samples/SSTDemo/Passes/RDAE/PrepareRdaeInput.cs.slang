#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"

const Texture2D<float4>  gAlbedo;
const Texture2D<float4>  gInColor;
const Texture2D<float4>  gInCNNAux;

RWStructuredBuffer<float3> gRdaeInput;
RWStructuredBuffer<float4> gRdaeAux;

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

    if (pixelPos.x >= gCNNDims.x || pixelPos.y >= gCNNDims.y)
        return;

    const int bufferIndex = pixelPos.y * gCNNDims.x + pixelPos.x;

    const float3 demodulated = gInColor[pixelPos].rgb / max(gAlbedo[pixelPos].rgb, float3(0.001, 0.001, 0.001));
    gRdaeInput[bufferIndex]  = pow(demodulated, gExponent);
    gRdaeAux[bufferIndex]    = gInCNNAux[pixelPos];
}
