#pragma once

#if defined(HOST_CODE)
#include "Falcor.h"
#endif

#ifndef HOST_CODE
#define SHADER_CODE
#endif

/** Returns a rotation matrix which aligns unit vector A with unit vector B. */
#ifdef HOST_CODE
float3x3 getRotationMatrixFromAToB(const float3& A, const float3& B)
#else
float3x3 getRotationMatrixFromAToB(in const float3 A, in const float3 B)
#endif
{
    const float3 v = cross(A, B);
    const float  c = dot(A, B);
    const float  f = 1.f / (1.f + c);
    const float xx = v.x * v.x;
    const float yy = v.y * v.y;
    const float zz = v.z * v.z;
    const float xy = v.x * v.y;
    const float yz = v.y * v.z;
    const float xz = v.x * v.z;

    // Special case: A = (-B)
    if (1.f + c == 0.f)
        return
    {
      -1.f,  0.f,  0.f, // row 1
       0.f, -1.f,  0.f, // row 2
       0.f,  0.f, -1.f, // row 3
    };

    return
    {
      1.f - (yy + zz) * f,          xy * f - v.z,        v.y + xz * f,  // Row 1
             v.z + xy * f,   1.f - (xx + zz) * f,        yz * f - v.x,  // Row 2
             xz * f - v.y,          yz * f + v.x, 1.f - (xx + yy) * f,  // Row 3
    };
}

/** Returns a rotation matrix which aligns the unit vector A with the z-axis. */
#ifdef HOST_CODE
float3x3 getRotMatToZAxis(const float3& A)
#else
float3x3 getRotMatToZAxis(in const float3 A)
#endif
{
    const float vx = A.y;
    const float vy = -A.x;
    const float c = A.z;
    const float s = sqrt(vx * vx + vy * vy);
    const float xx = vx * vx;
    const float yy = vy * vy;
    const float xy = vx * vy;
    const float  f = 1.f / (1.f + c);

    // Special case: A = (0, 0, -1)
    if (1.f + c == 0.f)
        return
    {
      -1.f,  0.f,  0.f, // row 1
       0.f, -1.f,  0.f, // row 2
       0.f,  0.f, -1.f, // row 3
    };

    return
    {
      1.f - yy * f,       xy * f,                  vy,  // Row 1
            xy * f, 1.f - xx * f,                 -vx,  // Row 2
               -vy,           vx, 1.f - (xx + yy) * f,  // Row 3
    };
}

#ifdef SHADER_CODE

/** Returns the squared length between vector A and B
*/
float lengthSq(in const float3 A, in const float3 B)
{
    return dot(A - B, A - B);
}

/** Returns true if ray intersects AABB, else false.
*/
bool intersectRayAABB(in float3 rayOrigin, in float3 rayDir, in float3 aabbMin, in float3 aabbMax, out float2 t)
{
    const float3 tMin = (aabbMin - rayOrigin) / rayDir;
    const float3 tMax = (aabbMax - rayOrigin) / rayDir;
    const float3 t1 = min(tMin, tMax);
    const float3 t2 = max(tMin, tMax);

    const float tNear = max(max(t1.x, t1.y), t1.z);
    const float tFar = min(min(t2.x, t2.y), t2.z);
    t = float2(tNear, tFar);
    return tNear > 0.f && tFar >= tNear;
}

/** Returns true if ray intersects plane, else false.
*/
bool intersectRayPlane(float3 planeP, float3 planeN, float3 rayP, float3 rayD, out float3 I)
{
    // Assuming vectors are all normalized
    float denom = dot(planeN, rayD);
    if (abs(denom) > 0.001f)
    {
        float3 p0l0 = planeP - rayP;
        float t = dot(p0l0, planeN) / denom;
        I = rayP + t * rayD;
        return (t >= 0);
    }
    return false;
}

/** Returns true if point P is within the AABB, else false.
*/
bool isPointWithinAABB(in const float3 P, in const float3 aabbMin, in const float3 aabbMax)
{
    return
        aabbMin.x < P.x  &&  P.x < aabbMax.x &&
        aabbMin.y < P.y  &&  P.y < aabbMax.y &&
        aabbMin.z < P.z  &&  P.z < aabbMax.z;
}

/** Returns an upper bound for the cosine between the normal of point P and an AABB.
*/
float maxNdotAABB(in const float3 P, in const float3 N, in const float3 aabbMin, in const float3 aabbMax)
{
    // Is point within the AABB? (with some epsilon due to numerical reasons...)
    const float3 epsilon = 0.001f;
    if (isPointWithinAABB(P, aabbMin - epsilon, aabbMax + epsilon))
        return 1.f;

    // Compute the upper cosine bound for the normal of point P to the AABB
    float3 zAxis = float3(0.f, 0.f, 1.f);
    bool flipZ = dot(N, zAxis) < 0.f;
    if (flipZ) zAxis.z = -zAxis.z;
    const float3x3 R = getRotationMatrixFromAToB(N, zAxis);

    // Transform AABB center
    const float3 hS = (aabbMax - aabbMin) * 0.5f;
    const float3 aaBBTCenter = mul(R, (aabbMin + hS) - P);

    const float3 Nx = R[0];
    const float3 Ny = R[1];
    const float3 Nz = R[2];

    float3 aabbTMin = float3(0.f);
    float3 aabbTMax = float3(0.f);

    // Compute each corner of the AABB and perfom min/max to get the extend
    const float3 v1 = Nx * hS.x + Ny * hS.y + Nz * hS.z;
    aabbTMin = min(aabbTMin, v1);
    aabbTMax = max(aabbTMax, v1);

    const float3 v2 = Nx * hS.x + Ny * hS.y + Nz * -hS.z;
    aabbTMin = min(aabbTMin, v2);
    aabbTMax = max(aabbTMax, v2);

    const float3 v3 = Nx * hS.x + Ny * -hS.y + Nz * hS.z;
    aabbTMin = min(aabbTMin, v3);
    aabbTMax = max(aabbTMax, v3);

    const float3 v4 = Nx * hS.x + Ny * -hS.y + Nz * -hS.z;
    aabbTMin = min(aabbTMin, v4);
    aabbTMax = max(aabbTMax, v4);

    const float3 v5 = Nx * -hS.x + Ny * hS.y + Nz * hS.z;
    aabbTMin = min(aabbTMin, v5);
    aabbTMax = max(aabbTMax, v5);

    const float3 v6 = Nx * -hS.x + Ny * hS.y + Nz * -hS.z;
    aabbTMin = min(aabbTMin, v6);
    aabbTMax = max(aabbTMax, v6);

    const float3 v7 = Nx * -hS.x + Ny * -hS.y + Nz * hS.z;
    aabbTMin = min(aabbTMin, v7);
    aabbTMax = max(aabbTMax, v7);

    const float3 v8 = Nx * -hS.x + Ny * -hS.y + Nz * -hS.z;
    aabbTMin = min(aabbTMin, v8);
    aabbTMax = max(aabbTMax, v8);

    aabbTMin += aaBBTCenter;
    aabbTMax += aaBBTCenter;

    const float zMax = flipZ ? -min(aabbTMin.z, aabbTMax.z) : max(aabbTMin.z, aabbTMax.z);
    if (zMax < 0.00001f) // Early out, cos would be <= 0
        return 0.f;

    float xMin, yMin;
    if ((aabbTMin.x < 0.f && aabbTMax.x > 0.f) || (aabbTMax.x < 0.f && aabbTMin.x > 0.f))
        xMin = 0.f;
    else
        xMin = min(abs(aabbTMin.x), abs(aabbTMax.x));

    if ((aabbTMin.y < 0.f && aabbTMax.y > 0.f) || (aabbTMax.y < 0.f && aabbTMin.y > 0.f))
        yMin = 0.f;
    else
        yMin = min(abs(aabbTMin.y), abs(aabbTMax.y));

    return zMax / length(float3(xMin, yMin, zMax)); // Div by zero not possible since zMax > 0
}

/** Returns the closet point on an AABB to a ray.
*/
float3 closestPointOnAABBRay(in float3 rayOrigin, in float3 rayDir, in float3 aabbMin, in float3 aabbMax)
{
    const float3 dirs[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };

    float dMin = FLT_MAX, dMax = -FLT_MAX, d;
    float3 minIs; // Will be written, since ray is guaranteed to intersect one of the planes!
    float3 Is;

    [unroll]
    for (int i = 0; i < 6; i++)
    {
        const float3 planeN = (i < 3) ? dirs[i % 3] : -dirs[i % 3];
        const float3 planeO = (i < 3) ? aabbMax : aabbMin;

        if (intersectRayPlane(planeO, planeN, rayOrigin, rayDir, Is))
        {
            Is = clamp(Is, aabbMin, aabbMax);
            float3 dirIs = normalize(Is - rayOrigin);
            d = dot(rayDir, dirIs);

            if (d > dMax)
            {
                dMin = d;
                dMax = d;
                minIs = Is;
            }
        }
    }
    return minIs;
}

#endif
