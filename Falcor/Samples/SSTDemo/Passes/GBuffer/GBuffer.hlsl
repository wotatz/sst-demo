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

__import Shading;
__import DefaultVS;

#include "HostDeviceSharedMacros.h"
#include "Passes/Shared/Packing.slang"

cbuffer PerFrameCB
{
  int2 gRenderTargetDim;
};

#ifndef INTERPOLATION_MODE
#define INTERPOLATION_MODE linear
#endif

struct GBufVertexOut
{
  VertexOut base;
  INTERPOLATION_MODE float3 normalObj : OBJECTNORMAL;
  uint instanceID : INSTANCEID;
};

// GBuffer
struct GBufferOut
{
  float4 PosW             : SV_Target0;  // world space position
  uint4  Packed1          : SV_Target1;  // world space normal, emissive color
  uint4  Packed2          : SV_Target2;  // material diffuse, opacity, material specular, roughness
  float4 CNNAux           : SV_Target3;  // CNN-specific buffer containing view space normal, depth, roughness
  float2 Motion           : SV_Target4;  // motion vector
  float2 PosNormalFwidth  : SV_Target5;
  float4 LinearZAndNormal : SV_Target6;
};

/**G-buffer rasterization vertex shader. */
GBufVertexOut vs(VertexIn vIn)
{
  GBufVertexOut vOut;
  vOut.base = defaultVS(vIn);       // Call the default Falcor vertex shader (see DefaultVS.slang)
  vOut.instanceID = vIn.instanceID; // Pass down the current instance ID for use in our G-buffer

#ifdef HAS_NORMAL
  vOut.normalObj = vIn.normal;      // Our g-buffer is storing an object-space normal, so pass that down
#else
  vOut.normalObj = 0;               //  .... Unless we don't have an object space normal.
#endif

  vOut.base.prevPosH.xy += vOut.base.prevPosH.w * 2 * float2(gCamera.jitterX, gCamera.jitterY);
  return vOut;
}

/**G-buffer rasterization pixel shader. */
GBufferOut ps(GBufVertexOut vsOut, float4 pixelCrd : SV_Position)
{
  ShadingData sd = prepareShadingData(vsOut.base, gMaterial, gCamera.posW);

  // The 'CNN Auxiliary' buffer
  float3 normV = mul(transpose(float3x3(gCamera.viewMat)), sd.N);
  float  depth = min(1.f, length(vsOut.base.posW - gCamera.posW) / (gCamera.farZ - gCamera.nearZ));
  float4 cnnAux = float4(normV.xy, depth, sd.linearRoughness);

  // Fill GBuffer
  GBufferOut gOut;
  gOut.PosW = float4(sd.posW, 1.f);

  packFloat2(sd.N.xy, gOut.Packed1.x);
  packFloat2(sd.N.zz, gOut.Packed1.y);
  packFloat2(sd.emissive.xy, gOut.Packed1.z);
  packFloat2(sd.emissive.zz, gOut.Packed1.w);

  packFloat2(sd.diffuse.xy, gOut.Packed2.x);
  packFloat2(float2(sd.diffuse.z, sd.opacity), gOut.Packed2.y);
  packFloat2(sd.specular.xy, gOut.Packed2.z);
  packFloat2(float2(sd.specular.z, sd.linearRoughness), gOut.Packed2.w);

  gOut.CNNAux          = cnnAux;
  gOut.Motion          = calcMotionVector(pixelCrd.xy, vsOut.base.prevPosH, gRenderTargetDim);
  gOut.PosNormalFwidth = float2(length(fwidth(sd.posW)), length(fwidth(sd.N)));

  float linearZ = vsOut.base.posH.z * vsOut.base.posH.w;
  float maxChangeZ = max(abs(ddx(linearZ)), abs(ddy(linearZ)));
  uint objNorm;
  packFloat2(ndir_to_oct_snorm(vsOut.normalObj), objNorm);
  gOut.LinearZAndNormal = float4(linearZ, maxChangeZ, vsOut.base.prevPosH.z, asfloat(objNorm));

  return gOut;
}
