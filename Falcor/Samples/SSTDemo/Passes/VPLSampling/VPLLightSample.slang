#pragma once

#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"
#include "../Shared/VPLData.h"

import Lights;

struct VPLLightSample
{
    float3 intensity; // The light intensity at the surface location
    float3 L;         // The direction from the surface to the light source
    float3 posW;      // The world-space position of the light
    float NdotH;      // Unclamped, can be negative
    float NdotL;      // Unclamped, can be negative
    float LdotH;      // Unclamped, can be negative
    float LdotLN;     // Unclamped, can be negative
    float distance;   // Distance from the light-source to the surface
    float falloff;    // Falloff from the light-source to the surface
};

inline VPLLightSample evalVPL(VPLData vpl, ShadingDataCompact sd)
{
    return evalVPL(vpl.getPosW(), vpl.getNormW(), vpl.getColor(), sd);
}

inline VPLLightSample evalVPL(float3 vplPosW, float3 vplNormW, float3 vplColor, ShadingDataCompact sd)
{
    VPLLightSample ls;
    ls.posW = vplPosW;
    ls.L = ls.posW - sd.posW;

    // Avoid NaN
    float distSquared = dot(ls.L, ls.L);
    ls.distance = (distSquared > 1e-5f) ? length(ls.L) : 0;
    ls.L = (distSquared > 1e-5f) ? normalize(ls.L) : 0;

    // Calculate the falloff
    ls.falloff = getDistanceFalloff(distSquared);

    ls.intensity = vplColor;
    ls.LdotLN = saturate(-dot(ls.L, vplNormW));

    // Compute common VPL properties
    float3 H = normalize(sd.V + ls.L);
    ls.NdotH = saturate(dot(sd.N, H));
    ls.NdotL = saturate(dot(sd.N, ls.L));
    ls.LdotH = saturate(dot(ls.L, H));

    return ls;
}
