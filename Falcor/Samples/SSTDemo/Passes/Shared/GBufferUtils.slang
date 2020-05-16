#include "Passes/Shared/Utils.slang"
#include "Passes/Shared/Packing.slang"

void unpackGBufPacked1(in float4 packed, out float3 normW, out float3 emissive)
{
    normW.xy = unpackFloat2(asuint(packed.x));
    normW.z = unpackFloat2(asuint(packed.y)).x;
    normW = normalize(normW);
    emissive.xy = unpackFloat2(asuint(packed.z));
    emissive.z = unpackFloat2(asuint(packed.w)).x;
}

void unpackGBufPacked2(in float4 packed, out float3 matDif, out float opacity, out float3 matSpec, out float linearRoughness)
{
    float2 tmp;
    matDif.xy = unpackFloat2(asuint(packed.x));
    tmp = unpackFloat2(asuint(packed.y));
    matDif.z = tmp.x;
    opacity = tmp.y;

    matSpec.xy = unpackFloat2(asuint(packed.z));
    tmp = unpackFloat2(asuint(packed.w));
    matSpec.z = tmp.x;
    linearRoughness = tmp.y;
}

float3 unpackGBufNormal(in uint4 packed1)
{
    float3 N;
    N.xy = unpackFloat2(packed1.x);
    N.z = unpackFloatHigh(packed1.y);
    return N;
}

float3 unpackGBufEmissive(in uint4 packed1)
{
    float3 E;
    E.xy = unpackFloat2(packed1.z);
    E.z = unpackFloatHigh(packed1.w);
    return E;
}
