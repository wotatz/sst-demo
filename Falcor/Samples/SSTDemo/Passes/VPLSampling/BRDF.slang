/**********************************************************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#  * Redistributions of code must retain the copyright notice, this list of conditions and the following disclaimer.
#  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
# SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************************************/

#pragma once

#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"

#include "VPLShadingData.slang"
#include "VPLLightSample.slang"


/** Get a GGX half vector / microfacet normal, sampled according to the GGX distribution
    When using this function to sample, the probability density is pdf = D * NdotH / (4 * HdotV)

    \param[in] u Uniformly distributed random numbers between 0 and 1
    \param[in] N Surface normal
    \param[in] roughness Roughness^2 of material
*/
float3 getGGXMicrofacet(float2 u, float3 N, float a2)
{
  float phi = M_PI2 * u.x;
  float cosTheta = sqrt(max(0, (1 - u.y)) / (1 + (a2 * a2 - 1) * u.y));
  float sinTheta = sqrt(max(0, 1 - cosTheta * cosTheta));

  // Tangent space H
  float3 tH;
  tH.x = sinTheta * cos(phi);
  tH.y = sinTheta * sin(phi);
  tH.z = cosTheta;

  float3 T = getPerpendicularVector(N);
  float3 B = normalize(cross(N, T));

  // World space H
  return normalize(T * tH.x + B * tH.y + N * tH.z);
}

// Our material has have both a diffuse and a specular lobe.  
//     With what probability should we sample the diffuse one?
float probabilityToSampleDiffuse(float3 difColor, float3 specColor)
{
  float lumDiffuse = max(0.01f, luminance(difColor.rgb));
  float lumSpecular = max(0.00f, luminance(specColor.rgb));
  return lumDiffuse / (lumDiffuse + lumSpecular);
}

float3 fresnelSchlick(float3 f0, float3 f90, float u)
{
  return f0 + (f90 - f0) * pow(1 - u, 5);
}

float ggxNormalDistribution(float a2, float NdotH)
{
  float d = ((NdotH * a2 - NdotH) * NdotH + 1);
  return a2 / max(0.01f, (d * d * M_PI));
}

float ggxNormalDistributionUnbounded(float a2, float NdotH)
{
  float d = ((NdotH * a2 - NdotH) * NdotH + 1);
  return a2 / (d * d * M_PI);
}

float evalSmithGGX(float NdotL, float NdotV, float a2)
{
  // Optimized version of Smith, already taking into account the division by (4 * NdotV * NdotL)
  //float a2 = roughness * roughness;

  // `NdotV *` and `NdotL *` are inversed. It's not a mistake.
  float ggxv = NdotL * sqrt((-NdotV * a2 + NdotV) * NdotV + a2);
  float ggxl = NdotV * sqrt((-NdotL * a2 + NdotL) * NdotL + a2);
  return 0.5f / (ggxv + ggxl);
}

float ggxSchlickMaskingTerm(float NdotL, float NdotV, float a2)
{
  // This from Schlick 1994, modified as per Karas in SIGGRAPH 2013 "Physically Based Shading" course
  // This function can be used for "G" in the Cook-Torrance model:  D*G*F / (4*NdotL*NdotV)

  // Karis notes they use alpha / 2 (or roughness^2 / 2)
  //float k = roughness * roughness / 2;
  float k = a2 / 2;

  // Karis also notes they can use the following equation, but only for analytical lights
  //float k = (roughness + 1)*(roughness + 1) / 8; 

  // Compute G(v) and G(l).  These equations directly from Schlick 1994
  //     (Though note, Schlick's notation is cryptic and confusing.)
  float g_v = NdotV / (NdotV*(1 - k) + k);
  float g_l = NdotL / (NdotL*(1 - k) + k);

  // Return G(v) * G(l)
  return g_v * g_l;
}

/** Lambertian diffuse
*/
float3 evalDiffuseLambertBrdf(ShadingDataCompact sd, VPLLightSample ls)
{
  return sd.diffuse.rgb * M_1_PI;
}

/** Evaluate diffuse BRDF
*/
float3 evalDiffuseBrdf(ShadingDataCompact sd, VPLLightSample ls)
{
  return evalDiffuseLambertBrdf(sd, ls);
}

/** Evaluate specular BRDF
*/
float3 evalSpecularBrdf(ShadingDataCompact sd, VPLLightSample ls)
{
  float a2 = sd.roughness * sd.roughness;

  float D = ggxNormalDistributionUnbounded(a2, ls.NdotH);
  float G = ggxSchlickMaskingTerm(ls.NdotL, sd.NdotV, a2);
  float3 F = fresnelSchlick(sd.specular, 1.f, max(0, ls.LdotH));
  float3 brdf = D * G * F / (4.f * sd.NdotV); // without NdotL!

  bool colorsNan = any(isnan(brdf));
  return colorsNan ? float3(0.f) : brdf;
}
