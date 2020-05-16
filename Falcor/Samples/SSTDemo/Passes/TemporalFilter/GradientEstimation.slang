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

#include "HostDeviceSharedCode.h"
#include "Passes/Shared/Utils.slang"
#include "Passes/Shared/Packing.slang"

const Texture2D gIllumination;
const Texture2D gPrevIllumination;
const Texture2D gAccHistory;

const Texture2D<float2> gMotion;
const Texture2D<float2> gPosNormalFwidth;
const Texture2D<float4> gLinearZAndNormal;
const Texture2D<float4> gPrevLinearZAndNormal;

struct PS_OUT
{
  float4  OutPrevIllumination : SV_TARGET0;
  float4  OutEstimation       : SV_TARGET1;
};

bool isReprjValid(int2 coord, float Z, float Zprev, float fwidthZ, float3 normal, float3 normalPrev, float fwidthNormal)
{
  const int2 imageDim = getTextureDims(gIllumination, 0);

  // check whether reprojected pixel is inside of the screen
  if (any(lessThan(coord, int2(0, 0))) || any(greaterThan(coord, imageDim - int2(1, 1))))
    return false;

  // check if deviation of depths is acceptable
  if (abs(Zprev - Z) / (fwidthZ + 1e-4) > 2.0)
      return false;

  // check normals for compatibility
  if (distance(normal, normalPrev) / (fwidthNormal + 1e-2) > 16.0)
    return false;

  return true;
}

float3 unpackNormal(float v)
{
    float2 packed = unpackFloat2(asuint(v));
    return oct_to_ndir_snorm(packed);
}

bool loadPrevData(float2 fragCoord, out float4 prevIllumination, out float historyLength)
{
  const int2 ipos = fragCoord;
  const float2 imageDim = float2(getTextureDims(gIllumination, 0));
  const float2 motion = gMotion[ipos];
  const float2 PosNormalFwidth = gPosNormalFwidth[ipos];

  // +0.5 to account for texel center offset
  const int2 iposPrev = int2(float2(ipos) + motion.xy * imageDim + float2(0.5, 0.5));

  // stores: Z, fwidth(z), z_prev
  float4 depth = gLinearZAndNormal[ipos];
  float3 normal = unpackNormal(depth.w);

  prevIllumination = float4(0, 0, 0, 0);

  bool v[4];
  const float2 posPrev = floor(fragCoord.xy) + motion.xy * imageDim;
  int2 offset[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
    
  // check for all 4 taps of the bilinear filter for validity
  bool valid = false;
  for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
  {
    int2 loc = (int2(posPrev) + offset[sampleIdx]);
    float4 depthPrev = gPrevLinearZAndNormal[loc];
    float3 normalPrev = unpackNormal(depthPrev.w);

    v[sampleIdx] = isReprjValid(iposPrev, depth.z, depthPrev.x, depth.y, normal, normalPrev, PosNormalFwidth.y);

    valid = valid || v[sampleIdx];
  }

  if (valid)
  {
    float sumw = 0;
    float x = frac(posPrev.x);
    float y = frac(posPrev.y);

    // bilinear weights
    float w[4] =
    {
      (1 - x) * (1 - y),
                            x * (1 - y),
                       (1 - x) * y,
                            x * y
    };

    prevIllumination = float4(0, 0, 0, 0);

    // perform the actual bilinear interpolation
    for (int sampleIdx = 0; sampleIdx < 4; sampleIdx++)
    {
      int2 loc = int2(posPrev) + offset[sampleIdx];
      if (v[sampleIdx])
      {
        prevIllumination += w[sampleIdx] * gPrevIllumination[loc];
        sumw += w[sampleIdx];
      }
    }

    // redistribute weights in case not all taps were used
    valid = (sumw >= 0.01);
    prevIllumination = valid ? prevIllumination / sumw : float4(0, 0, 0, 0);
  }
  if (!valid) // perform cross-bilateral filter in the hope to find some suitable samples somewhere
  {
    float cnt = 0.0;

    // this code performs a binary descision for each tap of the cross-bilateral filter
    const int radius = 1;
    for (int yy = -radius; yy <= radius; yy++)
    {
      for (int xx = -radius; xx <= radius; xx++)
      {
        int2 p = iposPrev + int2(xx, yy);
        float4 depthFilter = gPrevLinearZAndNormal[p];
        float3 normalFilter = unpackNormal(depthFilter.w);

        if (isReprjValid(iposPrev, depth.z, depthFilter.x, depth.y, normal, normalFilter, PosNormalFwidth.y))
        {
          prevIllumination += gPrevIllumination[p];
          cnt += 1.0;
        }
      }
    }
    if (cnt > 0)
    {
      valid = true;
      prevIllumination /= cnt;
    }

  }

  if (valid)
  {
    // crude, fixme
    historyLength = gAccHistory.Load(int3(iposPrev, 0)).r;
  }
  else
  {
    prevIllumination = float4(0, 0, 0, 0);
    historyLength = 0;
  }

  return valid;
}

float gatherLuminance(int2 ipos)
{
  const float2 motion = gMotion[ipos];
  const float2 PosNormalFwidth = gPosNormalFwidth[ipos];
  const float4 depth  = gLinearZAndNormal[ipos];
  const float3 normal = unpackNormal(depth.w);

  float lum = 0.f;
  float cnt = 0.0;

  const int radius = 3;
  for (int yy = -radius; yy <= radius; yy++)
  {
    for (int xx = -radius; xx <= radius; xx++)
    {
      int2 p = ipos + int2(xx, yy);
      float4 depthFilter  = gLinearZAndNormal[p];
      float3 normalFilter = unpackNormal(depthFilter.w);

      if (isReprjValid(ipos, depth.z, depthFilter.x, depth.y, normal, normalFilter, PosNormalFwidth.y) || (xx == 0 && yy == 0))
      {
        lum += luminance(gIllumination[p].rgb);
        cnt += 1.0;
      }
    }
  }

  return cnt > 0.f ? lum / cnt : 0.f;
}

PS_OUT main(float2 texC : TEXCOORD, float4 posH : SV_Position)
{
  const float4 fragCoord = posH;
  const int2 ipos = fragCoord.xy;
  const float4 color = gIllumination[ipos];

  // Load previous data
  float historyLength;
  float4 prevColor;
  bool success = loadPrevData(fragCoord.xy, prevColor, historyLength);
  historyLength = min(32.0f, success ? historyLength + 1.0f : 1.0f);

  const int radius = 1;
  bool neighborFail = false;
  for (int yy = -radius; yy <= radius; yy++)
  {
    for (int xx = -radius; xx <= radius; xx++)
    {
      int2 p = ipos + int2(xx, yy);
      float4 estimation = gAccHistory[p];
      if (estimation.z == 0)
      {
        neighborFail = true;
      }
    }
  }

  // Convert illumination to luminance
  const float lumIllumination     = gatherLuminance(ipos);
  const float lumPrevIllumination = luminance(prevColor.rgb);

  // Compute normalized illumnination gradient
  const float maxGradient     = max(lumIllumination, lumPrevIllumination);
  const float gradient        = lumIllumination - lumPrevIllumination;
  const float normedGradient  = clamp(abs(maxGradient > 1e-4 ? abs(gradient) / maxGradient : 0.f), 0.f, 1.f);

  // Write to rendertargets
  PS_OUT psOut;

  if (neighborFail)
    historyLength = 0.f;

  psOut.OutPrevIllumination = prevColor;
  psOut.OutEstimation       = float4(historyLength, normedGradient, success ? 1.f : 0.f, gradient);

  return psOut;
}
