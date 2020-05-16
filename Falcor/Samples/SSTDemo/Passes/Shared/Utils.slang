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

inline float3 getColorFromIntensity(float3 intensity)
{
    float mag = max(intensity.x, max(intensity.y, intensity.z));
    if (mag <= 1.f)
    {
        return intensity;
    }
    else
    {
        return intensity / mag;
    }
}

// Utility function to get a vector perpendicular to an input vector 
float3 getPerpendicularVector(float3 u)
{
    float3 a = abs(u);
    uint xm = ((a.x - a.y) < 0 && (a.x - a.z) < 0) ? 1 : 0;
    uint ym = (a.y - a.z) < 0 ? (1 ^ xm) : 0;
    uint zm = 1 ^ (xm | ym);
    return cross(u, float3(xm, ym, zm));
}

void getCoordinateSystem(in float3 v1, out float3 v2, out float3 v3)
{
  v2 = getPerpendicularVector(v1);
  v3 = cross(v1, v2);
}

int2 getTextureDims(Texture2D tex, uint mip)
{
  uint w, h;
  tex.GetDimensions(w, h);
  return int2(w, h);
}

int2 getTextureDims(RWTexture2D tex, uint mip)
{
  uint w, h;
  tex.GetDimensions(w, h);
  return int2(w, h);
}


#define COMPARE_FUNC2(TYPE) \
bool2 equal(TYPE a, TYPE b)              { return bool2(a.x == b.x, a.y == b.y); } \
bool2 notEqual(TYPE a, TYPE b)           { return bool2(a.x != b.x, a.y != b.y); } \
bool2 lessThan(TYPE a, TYPE b)           { return bool2(a.x <  b.x, a.y <  b.y); } \
bool2 greaterThan(TYPE a, TYPE b)        { return bool2(a.x >  b.x, a.y >  b.y); } \
bool2 lessThanEqual(TYPE a, TYPE b)      { return bool2(a.x <= b.x, a.y <= b.y); } \
bool2 greaterThanEqual(TYPE a, TYPE b)   { return bool2(a.x >= b.x, a.y >= b.y); }

COMPARE_FUNC2(int2)
COMPARE_FUNC2(float2)

#undef COMPARE_FUNC2

#define COMPARE_FUNC3(TYPE) \
bool3 equal(TYPE a, TYPE b)              { return bool3(a.x == b.x, a.y == b.y, a.z == b.z); } \
bool3 notEqual(TYPE a, TYPE b)           { return bool3(a.x != b.x, a.y != b.y, a.z != b.z); } \
bool3 lessThan(TYPE a, TYPE b)           { return bool3(a.x <  b.x, a.y <  b.y, a.z <  b.z); } \
bool3 greaterThan(TYPE a, TYPE b)        { return bool3(a.x >  b.x, a.y >  b.y, a.z >  b.z); } \
bool3 lessThanEqual(TYPE a, TYPE b)      { return bool3(a.x <= b.x, a.y <= b.y, a.z <= b.z); } \
bool3 greaterThanEqual(TYPE a, TYPE b)   { return bool3(a.x >= b.x, a.y >= b.y, a.z >= b.z); }

COMPARE_FUNC3(int3)
COMPARE_FUNC3(float3)

#undef COMPARE_FUNC3

#define COMPARE_FUNC4(TYPE) \
bool4 equal(TYPE a, TYPE b)              { return bool4(a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w); } \
bool4 notEqual(TYPE a, TYPE b)           { return bool4(a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w); } \
bool4 lessThan(TYPE a, TYPE b)           { return bool4(a.x <  b.x, a.y <  b.y, a.z <  b.z, a.w <  b.w); } \
bool4 greaterThan(TYPE a, TYPE b)        { return bool4(a.x >  b.x, a.y >  b.y, a.z >  b.z, a.w >  b.w); } \
bool4 lessThanEqual(TYPE a, TYPE b)      { return bool4(a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w); } \
bool4 greaterThanEqual(TYPE a, TYPE b)   { return bool4(a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w); }

COMPARE_FUNC4(int4)
COMPARE_FUNC4(float4)

#undef COMPARE_FUNC4
