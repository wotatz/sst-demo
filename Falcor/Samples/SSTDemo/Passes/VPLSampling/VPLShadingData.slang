#pragma once

struct ShadingDataCompact
{
    float3    posW;         ///< Shading hit position in world space
    float3    V;            ///< Direction to the eye at shading hit
    float3    N;            ///< Shading normal at shading hit
    float     NdotV;        // Unclamped, can be negative

    // Pre-loaded texture data
    float3 diffuse;
    float  opacity;
    float3 specular;
    float  linearRoughness; // This is the original roughness, before re-mapping. It is required for the Disney diffuse term
    float3 emissive;
    float  roughness;       // This is the re-mapped roughness value, which should be used for GGX computations
};
