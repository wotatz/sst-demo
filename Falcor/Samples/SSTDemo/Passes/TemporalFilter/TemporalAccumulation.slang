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

const Texture2D gIllumination;
const Texture2D gPrevIllumination;
const Texture2D gAccHistory;
const Texture2D gAlbedo;

struct PS_OUT
{
  float4 OutColor     : SV_TARGET0;
  float4 OutModulated : SV_TARGET1;
};

cbuffer PerFrameCB
{
  float gBaseExponent;
  bool  gModulate;
}


PS_OUT main(float2 texC : TEXCOORD, float4 posH : SV_Position)
{
  float4 fragCoord = posH;
  const int2 ipos = fragCoord.xy;

  const float3 color      = clamp(gIllumination[ipos].rgb, 0.f, 100.f); // CNN sometimes spits out weird values when no geometry comes into the screen
  const float3 prevColor  = gPrevIllumination[ipos].rgb;
  const float4 accHistory = gAccHistory[ipos];

  const float historyLength  = accHistory.x;
  const float normedGradient = accHistory.y;
  const bool  success        = accHistory.z > 0.f && historyLength > 0.f;
  const float gradient       = accHistory.w;

  const float exponent = lerp(1.f, gBaseExponent, pow((historyLength / 32.f), 2.f));

  float lambda = clamp(pow(normedGradient, exponent), 0.f, 1.f);

  const float gAlpha = 0.01f;
  const float temporal_alpha = success ? max(gAlpha, 1.0 / historyLength) : 1.0;
  float alpha_color = max(temporal_alpha, 1.0 / (historyLength + 1.0));

  alpha_color = lerp(alpha_color, 1.0, lambda);

  const float3 filtered = lerp(prevColor, color, alpha_color);

  PS_OUT psOut;
  psOut.OutColor     = float4(filtered, 1.f);
  psOut.OutModulated = float4(gModulate ? filtered * gAlbedo[ipos].rgb : filtered, 1.f);
  return psOut;
}
