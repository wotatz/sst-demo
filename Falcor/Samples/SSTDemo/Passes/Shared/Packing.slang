/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#pragma once

 // Include math constants (M_PI etc.). These are for use in this file only,
 // as macro definitions are not exported from a Slang module.
#include "HostDeviceSharedMacros.h"

 /******************************************************************************

     Octahedral mapping

     The center of the map represents the +z axis and its corners -z.
     The rotated inner square is the xy-plane projected onto the upper hemi-
     sphere, the outer four triangles folds down over the lower hemisphere.
     There are versions for equal-area and non-equal area (slightly faster).

     For details refer to:
     - Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD".
     - Cigolle et al. 2014, "Survey of Efficient Representations for Independent Unit Vectors".

 ******************************************************************************/

 /** Helper function to reflect the folds of the lower hemisphere
     over the diagonals in the octahedral map.
 */
float2 oct_wrap(float2 v)
{
    return (1.f - abs(v.yx)) * (v.xy >= 0.f ? 1.f : -1.f);
}

/** Converts normalized direction to the octahedral map (non-equal area, signed normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [-1,1] for each component.
*/
float2 ndir_to_oct_snorm(float3 n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane.
    float2 p = n.xy * (1.f / (abs(n.x) + abs(n.y) + abs(n.z)));
    p = (n.z < 0.f) ? oct_wrap(p) : p;
    return p;
}

/** Converts normalized direction to the octahedral map (non-equal area, unsigned normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [0,1] for each component.
*/
float2 ndir_to_oct_unorm(float3 n)
{
    return ndir_to_oct_snorm(n) * 0.5f + 0.5f;
}

/** Converts point in the octahedral map to normalized direction (non-equal area, signed normalized).
    \param[in] p Position in octahedral map in [-1,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_snorm(float2 p)
{
    float3 n = float3(p.xy, 1.0 - abs(p.x) - abs(p.y));
    n.xy = (n.z < 0.0) ? oct_wrap(n.xy) : n.xy;
    return normalize(n);
}

/** Converts point in the octahedral map to normalized direction (non-equal area, unsigned normalized).
    \param[in] p Position in octahedral map in [0,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_unorm(float2 p)
{
    return oct_to_ndir_snorm(p * 2.f - 1.f);
}

/** Converts normalized direction to the octahedral map (equal-area, unsigned normalized).
    \param[in] n Normalized direction.
    \return Position in octahedral map in [0,1] for each component.
*/
float2 ndir_to_oct_equal_area_unorm(float3 n)
{
    // Use atan2 to avoid explicit div-by-zero check in atan(y/x).
    float r = sqrt(1.f - abs(n.z));
    float phi = atan2(abs(n.y), abs(n.x));

    // Compute p = (u,v) in the first quadrant.
    float2 p;
    p.y = r * phi * M_2_PI;
    p.x = r - p.y;

    // Reflect p over the diagonals, and move to the correct quadrant.
    if (n.z < 0.f) p = 1.f - p.yx;
    p *= sign(n.xy);

    return p * 0.5f + 0.5f;
}

/** Converts point in the octahedral map to normalized direction (equal area, unsigned normalized).
    \param[in] p Position in octahedral map in [0,1] for each component.
    \return Normalized direction.
*/
float3 oct_to_ndir_equal_area_unorm(float2 p)
{
    p = p * 2.f - 1.f;

    // Compute radius r without branching. The radius r=0 at +z (center) and at -z (corners).
    float d = 1.f - (abs(p.x) + abs(p.y));
    float r = 1.f - abs(d);

    // Compute phi in [0,pi/2] (first quadrant) and sin/cos without branching.
    // TODO: Analyze fp32 precision, do we need a small epsilon instead of 0.0 here?
    float phi = (r > 0.f) ? ((abs(p.y) - abs(p.x)) / r + 1.f) * M_PI_4 : 0.f;

    // Convert to Cartesian coordinates. Note that sign(x)=0 for x=0, but that's fine here.
    float f = r * sqrt(2.f - r * r);
    float x = f * sign(p.x) * cos(phi);
    float y = f * sign(p.y) * sin(phi);
    float z = sign(d) * (1.f - r * r);

    return float3(x, y, z);
}

void packFloatHigh(float v, inout uint u)
{
    u &= 0x0000FFFF;
    u |= (f32tof16(v) << 16);
}

void packFloatLow(float v, inout uint u)
{
    u &= 0xFFFF0000;
    u |= f32tof16(v);
}

void packFloat2(float2 v, inout uint u)
{
    uint l = f32tof16(v.x);
    uint h = f32tof16(v.y);
    u = (h << 16) + l;
}

void packFloat3(float3 v, inout uint2 u)
{
    packFloat2(v.xy, u.x);
    packFloatHigh(v.z, u.y);
}

void packFloat4(float4 v, inout uint2 u)
{
    packFloat2(v.xy, u.x);
    packFloat2(v.yz, u.y);
}

float unpackFloatHigh(uint u)
{
    uint h = (u >> 16) & 0xffff;
    return f16tof32(h);
}

float unpackFloatLow(uint u)
{
    uint l = u & 0xffff;
    return f16tof32(l);
}

float2 unpackFloat2(uint u)
{
    uint l = u & 0xffff;
    uint h = (u >> 16) & 0xffff;
    return float2(f16tof32(l), f16tof32(h));
}

float3 unpackFloat3(uint2 u)
{
    return float3(unpackFloat2(u.x), unpackFloatHigh(u.y));
}

float4 unpackFloat4(uint2 u)
{
    return float4(unpackFloat2(u.x), unpackFloat2(u.y));
}
