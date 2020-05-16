
#include "HostDeviceData.h"
#include "../Shared/VPLData.h"

__import DefaultVS;

const StructuredBuffer<VPLData>  gVPLData;
const StructuredBuffer<VPLStats> gVPLStats;

cbuffer CB
{
  float4x4 gCamViewProjMat;
  float    gVPLRenderScale;
  float    gVPLColorScale;
}

VertexOut vs(VertexIn vIn, uint instanceID : SV_InstanceID)
{
    // Get vpl index
    const int indexOffset = 0;
    const int vplIndex = indexOffset + instanceID;

    const int numVPLS  = gVPLStats[0].numVPLs;
    const int numPaths = gVPLStats[0].numPaths;

    // Init vertices to be discarded in pixel shader
    VertexOut vOut;
    vOut.posH   = float4(0.f);
    vOut.colorV = float3(0.f);

    // Early out
    if (instanceID >= numVPLS)
        return vOut;

    // Assign vpl color
    vOut.colorV = gVPLData[vplIndex].getColor() * numPaths / gVPLColorScale;

    // Scale mesh vertices
    float4x4 transform = float4x4(
        float4(gVPLRenderScale, 0.f, 0.f, 0.f),
        float4(0.f, gVPLRenderScale, 0.f, 0.f),
        float4(0.f, 0.f, gVPLRenderScale, 0.f),
        float4(0.f, 0.f, 0.f, 1.f));

    // Transform vertices
    float4 posW = mul(transform, vIn.pos);
    posW.xyz   += gVPLData[vplIndex].getPosW();
    vOut.posW   = posW.xyz;
    vOut.posH   = mul(posW, gCamViewProjMat);

    return vOut;
}

float4 ps(VertexOut vsOut) : SV_TARGET
{
  if (vsOut.posH.w == 0.f)  discard;
  return float4(vsOut.colorV, 1.f);
}
