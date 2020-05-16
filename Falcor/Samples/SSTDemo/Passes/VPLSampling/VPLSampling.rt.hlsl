
#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"

#include "../Shared/VPLData.h"
#include "../Shared/VPLTreeStructs.h"
#include "../Shared/VPLUtils.h"
#include "VPLShading.slang"

#include "Passes/Shared/Utils.slang"
#include "Passes/Shared/GBufferUtils.slang"

import Lights;               // Light structures for our current scene
import Raytracing;           // Shared ray tracing specific functions & data
import ShaderCommon;         // Shared shading data structures
import Passes.Shared.Random;

AREA_LIGHTS

// VPL data
const shared RWStructuredBuffer<VPLData>  gVPLData;
const shared RWStructuredBuffer<VPLStats> gVPLStats;

// GBuffer
const shared Texture2D<float4> gPosW;
const shared Texture2D<float4> gPacked1;
const shared Texture2D<float4> gPacked2;

// Outputs
shared RWTexture2D<float4> gAlbedo;
shared RWTexture2D<float4> gCombined;
shared RWTexture2D<float4> gDirect;
shared RWTexture2D<float4> gIndirect;

// Sampling parameter constant buffer
shared cbuffer CB
{
    float gGMax;                  // Maxmimum value for bounded geometry term
    float gAttenuationEpsilon;    // Epsilon value for attenuation computation to guard against singularity.
    int   gMaxVPLs;               // Maximum number of VPLs
    int   gNumDirectSamples;      // Number of direct samples
    int   gNumIndirectSamples;    // Number of indirect samples
    int   gNumAccumulatedSamples; // Number of accumulated samples
};

// Per frame constant buffer
shared cbuffer PerFrameCB
{
    float gMinT;
    uint  gFrameCount;  // Frame counter used for random number generation
};


/** Sample a area light source intensity/direction at a shading point
*/
LightSample evalSampleAreaLight(in AreaLightData ld, in float3 surfacePosW, inout uint seed)
{
  const float3 dirW = mul(ld.dirW, ld.normMat).xyz;
  float2 rn = float2(nextRand(seed), nextRand(seed)) - 0.5f;
  const float3 lightPosW = mul(float4(ld.posW, 1.f), ld.transMat).xyz + mul(ld.bitangent, float3x3(ld.transMat)).xyz * rn.x * 1.f - mul(ld.tangent, float3x3(ld.transMat)).xyz * rn.y * 1.0f + dirW * 0.0001f;

  LightSample ls;
  ls.posW = lightPosW;
  ls.L = lightPosW - surfacePosW;
  float distSquared = dot(ls.L, ls.L);
  ls.distance = (distSquared > 1e-5f) ? length(ls.L) : 0;
  ls.L = (distSquared > 1e-5f) ? normalize(ls.L) : 0;

  // Calculate the falloff
  float cosTheta = dot(-ls.L, dirW);
  float falloff = max(0.f, cosTheta) * ld.scaledSurfaceArea;
  falloff *= getDistanceFalloff(distSquared);
  ls.diffuse = falloff * ld.intensity;
  ls.specular = ls.diffuse;
  return ls;
}

void getLightData(in int index, in float3 hitPos, out float3 toLight, out float3 lightIntensity, out float distToLight, inout uint seed)
{
  LightSample ls;

  // Is it a analytic light?
  if (index < gLightsCount)
  {
      // We only support point lights!
    ls = evalPointLight(gLights[index], hitPos);
  }
  else
  {
    // Must be an area light
    ls = evalSampleAreaLight(gAreaLights[index - gLightsCount], hitPos, seed);
  }

  // Convert the LightSample structure into simpler data
  toLight = normalize(ls.L);
  lightIntensity = ls.diffuse;
  distToLight = length(ls.posW - hitPos);
}

float3 sampleDirect(in const ShadingDataCompact sd, inout uint randSeed)
{
  // Pick a random light from our scene to sample for direct lighting
  const int numLights = gLightsCount + gAreaLightsCount;
  int lightToSample = min(int(nextRand(randSeed) * numLights), numLights - 1);

  // We need to query our scene to find info about the current light
  float distToLight;
  float3 lightIntensity;
  float3 L;
  getLightData(lightToSample, sd.posW, L, lightIntensity, distToLight, randSeed);

  // Compute our cosine / NdotL term
  float NdotL = saturate(dot(sd.N, L));

  // Shoot our ray for our direct lighting
  float shadowMult = float(numLights) * shootShadowRay(sd.posW + L * distToLight, sd.posW);
  float3 directColor = shadowMult * lightIntensity;
#ifdef USE_DIRECT_GGX
  // Compute half vectors and additional dot products for GGX
  float3 H = normalize(sd.V + L);
  float NdotH = saturate(dot(sd.N, H));
  float LdotH = saturate(dot(L, H));
  float NdotV = saturate(dot(sd.N, sd.V));

  // Evaluate terms for our GGX BRDF model
  float a2 = sd.roughness * sd.roughness;
  float  D = ggxNormalDistributionUnbounded(a2, NdotH);
  float  G = ggxSchlickMaskingTerm(NdotL, NdotV, a2);
  float3 F = fresnelSchlick(sd.specular, 1.f, max(0.f, LdotH));
  float3 ggxTerm = D * G * F / (4.f * NdotV);
  float3 directAlbedo = /*NdotL * */ ggxTerm + NdotL * sd.diffuse.rgb / M_PI; // ggx NdotL cancels out
#else
  float3 directAlbedo = NdotL * sd.diffuse.rgb / M_PI;
#endif
  float3 shadeColor = directColor * directAlbedo;

  return any(isnan(shadeColor)) ? float3(0.f) : shadeColor;
}

/** Shadow ray paylod. */
struct ShadowRayPayload
{
    float hit;
};

/** This function cast as shadow ray from origin to target. If origin is in shadow the function returns True. */ 
float shootShadowRay(float3 origin, float3 target)
{
  float3 direction = target - origin;
  float distance = length(direction);
  direction /= distance;

  RayDesc rayDesc;
  rayDesc.Origin = origin;
  rayDesc.Direction = direction;
  rayDesc.TMin = gMinT;
  rayDesc.TMax = distance - 0.0001f;

  ShadowRayPayload payload;
  payload.hit = 0.f;

  TraceRay(gRtScene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xFF, 0, hitProgramCount, 0, rayDesc, payload);
  return payload.hit;
}

[shader("anyhit")]
void shadowRayAnyHit(inout ShadowRayPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
  // Run a Falcor helper to extract the hit point's geometric data
  VertexOut  vsOut = getVertexAttributes(PrimitiveIndex(), attribs);

  // Extracts the diffuse color from the material (the alpha component is opacity)
  ExplicitLodTextureSampler lodSampler = { 0 };  // Specify the tex lod/mip to use here
  float4 baseColor = sampleTexture(gMaterial.resources.baseColor, gMaterial.resources.samplerState,
    vsOut.texC, gMaterial.baseColor, EXTRACT_DIFFUSE_TYPE(gMaterial.flags), lodSampler);

  // Test if this hit point passes a standard alpha test.  If not, discard/ignore the hit.
  if (baseColor.a < gMaterial.alphaThreshold)
  {
    IgnoreHit();
  }
  AcceptHitAndEndSearch();
}

[shader("miss")]
void shadowRayMiss(inout ShadowRayPayload payload)
{
    payload.hit = 1.f;
}

[shader("raygeneration")]
void rayGeneration()
{
  uint2 launchIndex = DispatchRaysIndex().xy;
  uint2 launchDim   = DispatchRaysDimensions().xy;

  // Load g-buffer data
  float4 posW    = gPosW[launchIndex];
  float4 packed1 = gPacked1[launchIndex];
  float4 packed2 = gPacked2[launchIndex];

  // Does this Gbuffer pixel contain a valid piece of geometry?  (0 in pos.w for invalid)
  bool isGeometryValid = (posW.w != 0.0f);

  float3 albedo        = float3(0.f);
  float3 directColor   = float3(0.f);
  float3 indirectColor = float3(0.f);

  // Do shading, if we have geoemtry here (otherwise, output the background color)
  if (isGeometryValid)
  {
    // Initialize a random number generator
    uint seed     = wang_hash((launchIndex.x + launchIndex.y * launchDim.x));
    uint randSeed = initRand(seed, gFrameCount, 32);

    // Prepare shading data
    ShadingDataCompact sd;
    sd.posW = posW.xyz;
    sd.V = normalize(gCamera.posW - posW.xyz);
    unpackGBufPacked1(packed1, sd.N, sd.emissive);
    unpackGBufPacked2(packed2, sd.diffuse, sd.opacity, sd.specular, sd.linearRoughness);
    sd.roughness = sd.linearRoughness * sd.linearRoughness;
    sd.NdotV = dot(sd.N, sd.V);

    // If this is an emissive surface we just return the emissive color
    bool isEmissive = any(sd.emissive);
    if (isEmissive)
    {
        directColor = getColorFromIntensity(sd.emissive);
    }
    else
    {
        // Sample direct contribution
#ifdef DIRECT_SAMPLING_ENABLED
        [unroll]
        for (int i = 0; i < gNumDirectSamples; i++)
            directColor += sampleDirect(sd, randSeed);
#endif
        // Sample indirect contribution
#ifdef INDIRECT_SAMPLING_ENABLED
        const int RootNodeIndex = gMaxVPLs; // Root node index is always the maximum number of VPLs!
        [unroll]
        for (int i = 0; i < gNumIndirectSamples; i++)
            indirectColor += sampleVPLs(sd, randSeed, RootNodeIndex, gVPLData, gVPLStats[0].numPaths, gVPLStats[0].numVPLs);
#endif

#ifdef ACCUMULATE_SAMPLES
        directColor   /= max(gNumDirectSamples, 1);
        indirectColor /= max(gNumIndirectSamples, 1);

        directColor   = (gNumAccumulatedSamples * gDirect[launchIndex].xyz + directColor) / (gNumAccumulatedSamples + 1);
        indirectColor = (gNumAccumulatedSamples * gIndirect[launchIndex].xyz + indirectColor) / (gNumAccumulatedSamples + 1);
#endif
    }
    albedo = sd.diffuse;
  }

  gAlbedo[launchIndex]   = float4(albedo, 1.f);
  gCombined[launchIndex] = float4(directColor + indirectColor, 1.f);
  gDirect[launchIndex]   = float4(directColor, 1.f);
  gIndirect[launchIndex] = float4(indirectColor, 1.f);
}


/** VPL sampling
*/
float3 sampleVPLs(in const ShadingDataCompact sd, inout uint randSeed, in const int rootNodeIdx, in const RWStructuredBuffer<VPLData> vplData, in const int numPaths, in const int numLeafs)
{
  if (numLeafs <= 0) return float3(0.f);

#if defined (USE_UNIFORM_SAMPLING)
  return sampleVPLArrayUniform(vplData, numLeafs, numPaths, sd, randSeed);
#else
  return sampleVPLTree(vplData, rootNodeIdx, numLeafs, numPaths, sd, randSeed);
#endif
}

/** Sample VPL uniformly from buffer
*/
float3 sampleVPLArrayUniform(in const RWStructuredBuffer<VPLData> vplData, in int numVPLs, in int numPaths, in ShadingDataCompact sd, inout uint rand_seed)
{
  const float p = 1.f / numVPLs;
  const int sampleID = int(numVPLs * nextRand(rand_seed));

#if defined(USE_INDIRECT_GGX)
  const float probDiffuse = probabilityToSampleDiffuse(sd.diffuse, sd.specular);
  const bool chooseDiffuse = (nextRand(rand_seed) < probDiffuse);
#else
  const bool chooseDiffuse = true;
#endif

  VPLLightSample ls = evalVPL(vplData[sampleID], sd);
  float visible = shootShadowRay(vplData[sampleID].getPosW(), sd.posW);
  return (visible > 0.f) ? evalVPL(ls, sd, gGMax, chooseDiffuse).rgb / p : float3(0.f);
}

/** Sample VPL from SST
*/
float3 sampleVPLTree(in const RWStructuredBuffer<VPLData> vplData, in const int rootIndex, in const int numVPLs, in const int numPaths, in ShadingDataCompact sd, inout uint rand_seed)
{
  // Root node index
  int parentIdx = rootIndex;
  int currentDepth = 0;

#if defined(USE_INDIRECT_GGX)
  const float probDiffuse  = probabilityToSampleDiffuse(sd.diffuse, sd.specular);
  const bool chooseDiffuse = (nextRand(rand_seed) < probDiffuse);
#else
  const float probDiffuse = 1.f;
  const bool chooseDiffuse = true;
#endif

  // Initialize the probability of picking the light
  float p = chooseDiffuse ? probDiffuse : (1.f - probDiffuse);
  float r = nextRand(rand_seed);

  // Get root node.
  VPLData vpl1 = vplData[parentIdx];

  while (true)
  {
    // Approximation good enough? Or did we reach a leaf node?
    if ((vpl1.getEarlyStop() > 0.f && chooseDiffuse) || vpl1.numVPLSubTree <= 0)
      break;

    // Get child nodes
    const int child1Id = vpl1.idChild1;
    const int child2Id = vpl1.idChild2;

    vpl1 = vplData[child1Id];
    VPLData vpl2 = vplData[child2Id];

    // Intensity term: I
    const float I1 = vpl1.getIntensity();
    const float I2 = vpl2.getIntensity();

    // Material term: M
    const float M1 = evalMaterial(sd, vpl1, currentDepth, numVPLs, chooseDiffuse, rand_seed);
    const float M2 = evalMaterial(sd, vpl2, currentDepth, numVPLs, chooseDiffuse, rand_seed);

    // Geometric term: G
    const float G1 = 1.f; // Omni
    const float G2 = 1.f; // Omni

    // Attenuation term: A
    const float A1 = evalAttenuation(sd, vpl1);
    const float A2 = evalAttenuation(sd, vpl2);

    // Compute importance weights
    const float w1 = G1 * M1 * A1 * I1;
    const float w2 = G2 * M2 * A2 * I2;

    // Select child node
    if (w1 + w2 > 0)
    {
      const float p1 = w1 / (w1 + w2);
      if (r <= p1)
      {
        p = p * p1;
        r = r / p1;
        parentIdx = child1Id;
      }
      else
      {
        p = p * (1.f - p1);
        r = (r - p1) / (1.f - p1);
        parentIdx = child2Id;
        vpl1 = vpl2;
      }
      currentDepth++;
    }
    else
    {
      // Quit stochastic traversal => dead branch
      return 0.f;
    }
  }

  // Get position on plane and sample
  VPLData vpl = vplData[parentIdx];
  const float3 samplePosW = normalPointOnPlane(vpl.getNormW(), vpl.getPosW(), vpl.getVariance(), vpl.getAABBMin(), vpl.getAABBMax(), rand_seed);
  VPLLightSample ls = evalVPL(samplePosW, vpl.getNormW(), vpl.getColor(), sd);

  float visible = shootShadowRay(ls.posW, sd.posW);
  return (p > 0.f && visible > 0.f) ? evalVPL(ls, sd, gGMax, chooseDiffuse).rgb / p : float3(0.f);
}

float evalMaterial(in ShadingDataCompact sd, in VPLData vpl, in int depth, in int vplNum, in bool chooseDiffuse, inout uint randSeed)
{
    VPLLightSample ls = evalVPL(vpl, sd);
    const float brdf = evalBrdf(sd, ls, vpl, chooseDiffuse, depth, vplNum, randSeed);
    return brdf * maxNdotAABB(sd.posW, sd.N, vpl.getAABBMin(), vpl.getAABBMax());
}

float evalAttenuation(in ShadingDataCompact sd, in VPLData vpl)
{
    return 1.f / max(lengthSq(sd.posW, vpl.getPosW()), gAttenuationEpsilon);
}

float evalBrdf(in ShadingDataCompact sd, in VPLLightSample ls, in VPLData vpl, in bool chooseDiffuse, in int depth, in int vplNum, inout uint randSeed)
{
#ifdef USE_INDIRECT_GGX
    if (chooseDiffuse)
    {
        return max(computeIntensity(evalDiffuseBrdf(sd, ls)), 0.01f);
    }
    else
    {
        // Heuristic for estimating the glossy brdf term:
        // Find the closest point on AABB with regard to the perfect reflection.
        float3 Lp = reflect(-sd.V, sd.N);
        float3 L = Lp;
        float3 Is = vpl.getPosW();

        const float3 epsilon = 0.001f;
        if (isPointWithinAABB(sd.posW, vpl.getAABBMin() - epsilon, vpl.getAABBMax() + epsilon))
            L = Lp;
        else
        {
            Is = closestPointOnAABBRay(sd.posW, Lp, vpl.getAABBMin(), vpl.getAABBMax());
            L = normalize(Is - sd.posW);
        }
    
        float3 H = normalize(sd.V + L);
        float NdotL = saturate(dot(sd.N, L));
        float NdotH = saturate(dot(sd.N, H));
        float LdotH = saturate(dot(L, H));
        float NdotV = saturate(dot(sd.N, sd.V));

        float a2 = sd.roughness * sd.roughness;
        float  D = ggxNormalDistributionUnbounded(a2, NdotH);
        float  G = ggxSchlickMaskingTerm(NdotL, NdotV, a2);
        float3 F = fresnelSchlick(sd.specular, 1.f, max(0.f, LdotH));
        float3 ggxTerm = D * G * F / (4.f * NdotV * NdotL);
        ggxTerm = any(isnan(ggxTerm)) ? float3(0.f) : ggxTerm;

        return computeIntensity(ggxTerm);
    }
#else
    return max(computeIntensity(evalDiffuseBrdf(sd, ls)), 0.01f);
#endif
}

inline float computeIntensity(float3 rgb)
{
    return dot(rgb, float3(0.299f, 0.587f, 0.114f));
}

/** Returns normal distributed point on plane bounded by an AABB. */
float3 normalPointOnPlane(in const float3 N, in const float3 O, in const float3 variance, in const float3 aabbMin, in const float3 aabbMax, inout uint randSeed)
{
  const float2   xy = nextNormal2(float2(0.f), sqrt(variance.xy), randSeed);
  const float3x3 R  = getRotationMatrixFromAToB(N, float3(0.f, 0.f, 1.f));
  const float3   P  = O + R[0] * xy.x + R[1] * xy.y;
  return clamp(P, aabbMin, aabbMax);
}
