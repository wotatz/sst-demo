#pragma once

#include "HostDeviceSharedMacros.h"

uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

uint initRand(uint val0, uint val1, uint backoff = 16)
{
    uint v0 = val0, v1 = val1, s0 = 0;

    [unroll]
    for (uint n = 0; n < backoff; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

float nextRand(inout uint s)
{
    s = (1664525u * s + 1013904223u);
    return float(s & 0x00FFFFFF) / float(0x01000000);
}

float2 nextRand2(inout uint s)
{
    return float2(nextRand(s), nextRand(s));
}

float3 nextRand3(inout uint s)
{
    return float3(nextRand(s), nextRand(s), nextRand(s));
}

float4 nextRand4(inout uint s)
{
    return float4(nextRand(s), nextRand(s), nextRand(s), nextRand(s));
}

float boxMuller(in const float mean, in const float std, inout uint rngSeed)
{
    const float u0 = nextRand(rngSeed);
    const float u1 = nextRand(rngSeed);
    const float r = sqrt(-2.f * log(u0));
    const float theta = 2.f * M_PI * u1;
    return r * sin(theta) * std + mean;
}

float2 boxMuller(in const float mean1, in const float std1, in const float mean2, in const float std2, inout uint rngSeed)
{
    const float u0 = nextRand(rngSeed);
    const float u1 = nextRand(rngSeed);
    const float r = sqrt(-2.f * log(u0));
    const float theta = 2.f * M_PI * u1;
    return float2(r * sin(theta) * std1 + mean1, r * cos(theta) * std2 + mean2);
}

float nextNormal(in const float mean, in const float std, inout uint s)
{
    return boxMuller(mean, std, s);
}

float2 nextNormal2(in const float2 mean, in const float2 std, inout uint s)
{
    return boxMuller(mean.x, std.x, mean.y, std.y, s);
}

float3 nextNormal3(in const float3 mean, in const float3 std, inout uint s)
{
    return float3(nextNormal2(mean.xy, std.xy, s), nextNormal(mean.z, std.z, s));
}

float3 perp_stark(float3 u)
{
    float3 a = abs(u);
    uint uyx = (a.x - a.y) < 0 ? 1 : 0;
    uint uzx = (a.x - a.z) < 0 ? 1 : 0;
    uint uzy = (a.y - a.z) < 0 ? 1 : 0;
    uint xm = uyx & uzx;
    uint ym = (1 ^ xm) & uzy;
    uint zm = 1 ^ (xm | ym);  // 1 ^ (xm & ym)
    float3 v = cross(u, float3(xm, ym, zm));
    return v;
}

/** Helper function to transform vector v from the local surface frame to world space.
*/
float3 fromLocal(float3 v, float3 N)
{
    float3 B = perp_stark(N);
    float3 T = cross(B, N);
    return T * v.x + B * v.y + N * v.z;
}

float3 uniformSphereSample(in float3 N, in float2 rn)
{
    const float z = 1.f - 2.f * rn.x;
    const float r = sqrt(max(0.f, 1.f - z * z));
    const float phi = 2 * M_PI * rn.y;
    return fromLocal(float3(r * cos(phi), r * sin(phi), z), N);
}

float3 cosineHemisphereSample(float3 N, float2 rn)
{
    const float r = sqrt(rn.x);
    const float phi = rn.y * M_PI2;
    float3 L = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0f, 1.0f - rn.x)));
    return fromLocal(L, N);
}
