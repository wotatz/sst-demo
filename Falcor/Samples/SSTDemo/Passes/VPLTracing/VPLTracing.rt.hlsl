
#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"

#include "Passes/Shared/VPLData.h"

import Raytracing;                   // Shared ray tracing specific functions & data
import ShaderCommon;                 // Shared shading data structures
import Shading;                      // Shading functions, etc     
import Lights;                       // Light structures for our current scene
import Passes.Shared.Random;

shared RWStructuredBuffer<VPLData>  gVPLData;
shared RWStructuredBuffer<VPLStats> gVPLStats;
shared RWStructuredBuffer<float3>   gVPLPositions;

#include "Passes/Shared/Utils.slang"

AREA_LIGHTS

shared cbuffer CB
{
  LightInfo gLightInfos[NUM_LIGHT_SOURCES];
  float  gMinT;
  uint   gFrameCount;
  int    gNumMaxBounces;
  int    gNumMinBounces;
  int    gNumPaths;
};

struct VPLrayLoad
{
    float3  radiance;
    float   q;
    int     bounces;
    uint    rngState;
};

LightInfo determineLightInfo(in const uint rayIndex)
{
  uint lastLaunchRange = 0;
  [unroll]
  for (int i = 0; i < NUM_LIGHT_SOURCES; i++)
  {
    const uint currentLaunchRange = gLightInfos[i].rayRange.y;
    if (rayIndex < currentLaunchRange)
    {
      return gLightInfos[i];
    }
    lastLaunchRange = currentLaunchRange;
  }
}

[shader("raygeneration")]
void VPLTracingRayGen()
{
  uint2 launchIndex = DispatchRaysIndex().xy;
  uint2 launchDim = DispatchRaysDimensions().xy;

  LightInfo li = determineLightInfo(launchIndex.x);
  const uint numRays = li.rayRange.y - li.rayRange.x;
  const uint numTotalRays = DispatchRaysDimensions().x;
  const float rayRatio = (float) numRays / numTotalRays; // This accounts for distribution of rays for lightsources based on intensity

  // Prepare rng
  uint seed = wang_hash(launchIndex.x);
  uint rngState = initRand(seed, gFrameCount, 16);

  // Setup ray description
  RayDesc rayDesc;
  rayDesc.TMin = gMinT;
  rayDesc.TMax = FLT_MAX;

  // Setup rayload
  VPLrayLoad rayLoad;
  rayLoad.rngState = rngState;
  rayLoad.bounces  = 0;
  rayLoad.q        = 1.f;  // Path start has survivabilty of 100%

  // Which type of light?
  if (li.type == LightPoint)
  {
      LightData ld = gLights[li.index];
      const float pdf = 1.f / M_PI;
      const float3 radiance = ld.intensity / pdf / rayRatio / gNumPaths;

      rayDesc.Origin    = ld.posW;
      rayDesc.Direction = uniformSphereSample(float3(1,0,0), nextRand2(rayLoad.rngState));
      rayLoad.radiance  = radiance;
  }
  else if (li.type == LightArea)
  {
      AreaLightData ld = gAreaLights[li.index];
      const float pdf = 1.f / ld.scaledSurfaceArea;
      const float3 radiance = ld.intensity / pdf / rayRatio / gNumPaths;

      // Note: Tangent and bitangent are the actual extents! We assume that all area light sources are rectangular and consist of only 2 triangles!
      float2 rn = float2(nextRand(rayLoad.rngState), nextRand(rayLoad.rngState)) - 0.5f;
      const float3 dirL = mul(ld.dirW, ld.normMat).xyz;
      rayDesc.Origin = mul(float4(ld.posW,1.f), ld.transMat).xyz + mul(ld.bitangent, float3x3(ld.transMat)).xyz * rn.x - mul(ld.tangent, float3x3(ld.transMat)).xyz * rn.y + dirL * 0.0001f;
      rayDesc.Direction = cosineHemisphereSample(dirL, float2(nextRand(rayLoad.rngState), nextRand(rayLoad.rngState)));
      rayLoad.radiance = radiance;
  }
  else
      return;

  // Increment VPL path counter (should match numPaths)
  incrementVPLPaths();

  if (gNumMaxBounces <= 0)
    return;

  // Start VPL ray
  TraceRay(gRtScene, 0, 0xFF, 0, hitProgramCount, 0, rayDesc, rayLoad);
}

[shader("miss")]
void VPLTraceMiss(inout VPLrayLoad rayLoad : SV_RayPayload)
{
}

[shader("closesthit")]
void VPLTraceClosestHit(inout VPLrayLoad rayLoad, in BuiltInTriangleIntersectionAttributes attribs)
{
    // Get some hit information
    VertexOut vsOut = getVertexAttributes(PrimitiveIndex(), attribs);
    ShadingData sd = prepareShadingData(vsOut, gMaterial, WorldRayOrigin(), 0);
    float rayLength = RayTCurrent();

    // Compute tangent space if it is invalid.
    if (!(dot(sd.B, sd.B) > 0.f))
    {
        sd.B = perp_stark(sd.N);
        sd.T = cross(sd.B, sd.N);
    }

    // Reject VPLs which hit back side of triangles
    if (dot(-WorldRayDirection(), sd.N) < 0)
        return;

    // Calculate survival probability (russian roulette) which is proportional to the albedo of the current surface
    // ... but only if the minimum number of bounces has been reached
    const float q = rayLoad.bounces > gNumMinBounces ? max(sd.diffuse.x, max(sd.diffuse.y, sd.diffuse.z)) : 1.f;

    // Adjust VPL ray payload
    rayLoad.radiance *= sd.diffuse / rayLoad.q;
    rayLoad.q         = q;
    rayLoad.bounces  += 1;

    // Place VPL at current position
    addVPL(gVPLData, gVPLPositions, getNextID(), sd.posW, sd.N, rayLoad.radiance);

    // Maximum bounces reached
    if (rayLoad.bounces >= gNumMaxBounces)
        return;

    // Russian Roulette - Is the chamber loaded? ;)
    const float r = nextRand(rayLoad.rngState);
    if (r < (1 - q))
        return;

    // Next bounce
    RayDesc rayDesc;
    rayDesc.Origin    = sd.posW;
    rayDesc.Direction = cosineHemisphereSample(sd.N, float2(nextRand(rayLoad.rngState), nextRand(rayLoad.rngState)));
    rayDesc.TMin      = gMinT;
    rayDesc.TMax      = 1.0e38f;
    TraceRay(gRtScene, 0, 0xFF, 0, hitProgramCount, 0, rayDesc, rayLoad);
}

int getNextID()
{
  int nextId = 0;
  InterlockedAdd(gVPLStats[0].numVPLs, 1, nextId); // Increment VPL counter
  return nextId;
}

int incrementVPLPaths()
{
  int numPaths = 0;
  InterlockedAdd(gVPLStats[0].numPaths, 1, numPaths); // Increment path counter
  return numPaths;
}

void addVPL(in RWStructuredBuffer<VPLData> vplData, in RWStructuredBuffer<float3> vplPositions, in int id, in float3 posW, in float3 normW, in float3 radiance)
{
  VPLData vpl;
  vpl.setPosW(posW);
  vpl.setEarlyStop(0.f);

  vpl.setNormW(normW);
  vpl.setColor(radiance);
  const float intensity = luminance(radiance);
  vpl.setIntensity(intensity);

  vpl.setAABBMin(posW);
  vpl.setAABBMax(posW);
  vpl.setVariance(float3(0.f));

  vpl.id = id;
  vpl.idChild1 = -1;
  vpl.idChild2 = -1;
  vpl.numVPLSubTree = 0;

  // Write data to buffers
  vplData[vpl.id]      = vpl;
  vplPositions[vpl.id] = posW;
}
