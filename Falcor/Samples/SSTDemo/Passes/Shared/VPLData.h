#pragma once

#ifndef VPL_DATA
#define VPL_DATA

#if defined(HOST_CODE)
    #include "Falcor.h"
    #define DEFAULTS(x_) = x_
    #define CONST_MOD const
    #define MUTATING
    using uint2 = glm::uvec2;
    using uint = unsigned int;
#else
    #define SHADER_CODE
    #include "Packing.slang"
    #define DEFAULTS(x_)
    #define CONST_MOD
    #define MUTATING [mutating]
#endif

struct VPLStats
{
    int numVPLs     DEFAULTS(0);
    int numPaths    DEFAULTS(0);
    int pad2        DEFAULTS(-1);
    int pad3        DEFAULTS(-1);
};

struct LightInfo
{
    int  type     DEFAULTS(-1);
    int  index    DEFAULTS(-1);
    int2 rayRange DEFAULTS(int2(-1, -1));
};

struct VPLData
{
    uint2 posW;
    uint2 normW;
    // 16

    uint2 aabbMin;
    uint2 aabbMax;
    //16

    uint2 rad;
    uint2 var;
    //16

    int id;
    int idChild1;
    int idChild2;
    int numVPLSubTree;
    //16

    /** Getter methods
    */

    inline float3 getPosW() CONST_MOD
    {
#if defined(HOST_CODE)
        return float3(glm::unpackHalf2x16(posW.x), glm::unpackHalf2x16(posW.y).y);
#else
        return unpackFloat3(posW);
#endif
    }

    inline float3 getNormW() CONST_MOD
    {
#if defined(HOST_CODE)
        return float3(glm::unpackHalf2x16(normW.x), glm::unpackHalf2x16(normW.y).y);
#else
        return unpackFloat3(normW);
#endif
    }

    inline float3 getColor() CONST_MOD
    {
#if defined(HOST_CODE)
        return float3(glm::unpackHalf2x16(rad.x), glm::unpackHalf2x16(rad.y).y);
#else
        return unpackFloat3(rad);
#endif
    }

    inline float getIntensity() CONST_MOD
    {
#if defined(HOST_CODE)
        return glm::unpackHalf2x16(rad.y).x;
#else
        return unpackFloatLow(rad.y);
#endif
    }

    inline float3 getAABBMin() CONST_MOD
    {
#if defined(HOST_CODE)
        return float3(glm::unpackHalf2x16(aabbMin.x), glm::unpackHalf2x16(aabbMin.y).y);
#else
        return unpackFloat3(aabbMin);
#endif
    }

    inline float3 getAABBMax() CONST_MOD
    {
#if defined(HOST_CODE)
        return float3(glm::unpackHalf2x16(aabbMax.x), glm::unpackHalf2x16(aabbMax.y).y);
#else
        return unpackFloat3(aabbMax);
#endif
    }

    inline float3 getVariance() CONST_MOD
    {
#if defined(HOST_CODE)
        return float3(glm::unpackHalf2x16(var.x), glm::unpackHalf2x16(var.y).y);
#else
        return unpackFloat3(var);
#endif
    }

    inline float getEarlyStop() CONST_MOD
    {
#if defined(HOST_CODE)
        return glm::unpackHalf2x16(var.y).x;
#else
        return unpackFloatLow(var.y);
#endif
    }

#ifdef SHADER_CODE

    /** Setter methods
    */

    MUTATING inline void setPosW(float3 p)
    {
        packFloat3(p, posW);
    }

    MUTATING inline void setNormW(float3 n)
    {
         packFloat3(n, normW);
    }

    MUTATING inline void setColor(float3 c)
    {
         packFloat3(c, rad);
    }

    MUTATING inline void setIntensity(float i)
    {
        packFloatLow(i, rad.y);
    }

    MUTATING inline void setAABBMin(float3 min)
    {
         packFloat3(min, aabbMin);
    }

    MUTATING inline void setAABBMax(float3 max)
    {
         packFloat3(max, aabbMax);
    }

    MUTATING inline void setVariance(float3 variance)
    {
        packFloat3(variance, var);
    }

    MUTATING inline void setEarlyStop(float f)
    {
        packFloatLow(f, var.y);
    }
#endif
};

#endif
