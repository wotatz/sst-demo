#pragma once

#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"
#include "../Shared/VPLData.h"

#include "VPLShadingData.slang"
#include "VPLLightSample.slang"
#include "BRDF.slang"


float3 evalIntensity(VPLLightSample ls, ShadingDataCompact sd, float GMax)
{
    return ls.intensity * min(ls.LdotLN * ls.falloff, GMax);
}

float4 evalDiffuse(VPLLightSample ls, ShadingDataCompact sd)
{
    float3 diffuseBrdf = ls.NdotL * evalDiffuseBrdf(sd, ls);
    float3 diffuse = ls.intensity * diffuseBrdf;
    return float4(diffuse, sd.opacity);
}

float3 evalSpecular(VPLLightSample ls, ShadingDataCompact sd)
{
    float3 specularBrdf = evalSpecularBrdf(sd, ls); // NdotL cancels out
    float3 specular = ls.intensity * specularBrdf;
    return specular;
}

float4 evalVPL(VPLLightSample ls, ShadingDataCompact sd, float GMax, bool chooseDiffuse)
{
  // If the light doesn't hit the surface or we are viewing the surface from the back, return
  if (ls.NdotL <= 0) return float4(0.f, 0.f, 0.f, 1.f);
  sd.NdotV = saturate(sd.NdotV);

  float4 color;
  ls.intensity = evalIntensity(ls, sd, GMax);

  if (chooseDiffuse)
  {
    color = evalDiffuse(ls, sd);
  }
  else
  {
    color = float4(evalSpecular(ls, sd), 1.f);
  }

  return color;
}
