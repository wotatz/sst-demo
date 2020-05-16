/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#ifndef _HOST_DEVICE_SHARED_MACROS_H
#define _HOST_DEVICE_SHARED_MACROS_H

/*******************************************************************
                    Common structures & routines
*******************************************************************/

#define MAX_INSTANCES 64    ///< Max supported instances per draw call
#define MAX_BONES 256       ///< Max supported bones per model

/*******************************************************************
                    Glue code for CPU/GPU compilation
*******************************************************************/

#if (defined(__STDC_HOSTED__) || defined(__cplusplus))   // we're in C-compliant compiler, probably host
#define HOST_CODE 1
#else
#define FALCOR_SHADER_CODE
#endif


#ifdef HOST_CODE

/*******************************************************************
                    CPU declarations
*******************************************************************/
#define DEFAULTS(x_) = x_
#define SamplerState std::shared_ptr<Sampler>
#define Texture2D std::shared_ptr<Texture>
#else
/*******************************************************************
                    HLSL declarations
*******************************************************************/
#define inline 
#define DEFAULTS(x_)
#endif

/*******************************************************************
    Materials
*******************************************************************/

// Shading model
#define ShadingModelMetalRough 0
//#define ShadingModelMetalAnisoRough 1 Reserved for future use
#define ShadingModelSpecGloss 2

// Channel type
#define ChannelTypeUnused    0
#define ChannelTypeConst     1
#define ChannelTypeTexture   2

// Normal map type
#define NormalMapUnused     0
#define NormalMapRGB        1
#define NormalMapRG         2

// Alpha mode
#define AlphaModeOpaque      0
#define AlphaModeMask        1

// Bit count
#define SHADING_MODEL_BITS   (3)
#define DIFFUSE_TYPE_BITS    (3)
#define SPECULAR_TYPE_BITS   (3)
#define EMISSIVE_TYPE_BITS   (3)
#define NORMAL_MAP_BITS      (2)
#define OCCLUSION_MAP_BITS   (1)
#define LIGHT_MAP_BITS       (1)
#define HEIGHT_MAP_BITS      (1)
#define ALPHA_MODE_BITS      (2)
#define DOUBLE_SIDED_BITS    (1)

// Offsets
#define SHADING_MODEL_OFFSET (0)
#define DIFFUSE_TYPE_OFFSET  (SHADING_MODEL_OFFSET + SHADING_MODEL_BITS)
#define SPECULAR_TYPE_OFFSET (DIFFUSE_TYPE_OFFSET  + DIFFUSE_TYPE_BITS)
#define EMISSIVE_TYPE_OFFSET (SPECULAR_TYPE_OFFSET + SPECULAR_TYPE_BITS)
#define NORMAL_MAP_OFFSET    (EMISSIVE_TYPE_OFFSET + EMISSIVE_TYPE_BITS)
#define OCCLUSION_MAP_OFFSET (NORMAL_MAP_OFFSET    + NORMAL_MAP_BITS)
#define LIGHT_MAP_OFFSET     (OCCLUSION_MAP_OFFSET + OCCLUSION_MAP_BITS)
#define HEIGHT_MAP_OFFSET    (LIGHT_MAP_OFFSET     + LIGHT_MAP_BITS)
#define ALPHA_MODE_OFFSET    (HEIGHT_MAP_OFFSET    + HEIGHT_MAP_BITS)
#define DOUBLE_SIDED_OFFSET  (ALPHA_MODE_OFFSET    + ALPHA_MODE_BITS)

// Extract bits
#define EXTRACT_BITS(bits, offset, value) ((value >> offset) & ((1 << bits) - 1))
#define EXTRACT_SHADING_MODEL(value)    EXTRACT_BITS(SHADING_MODEL_BITS,    SHADING_MODEL_OFFSET,   value)
#define EXTRACT_DIFFUSE_TYPE(value)     EXTRACT_BITS(DIFFUSE_TYPE_BITS,     DIFFUSE_TYPE_OFFSET,    value)
#define EXTRACT_SPECULAR_TYPE(value)    EXTRACT_BITS(SPECULAR_TYPE_BITS,    SPECULAR_TYPE_OFFSET,   value)
#define EXTRACT_EMISSIVE_TYPE(value)    EXTRACT_BITS(EMISSIVE_TYPE_BITS,    EMISSIVE_TYPE_OFFSET,   value)
#define EXTRACT_NORMAL_MAP_TYPE(value)  EXTRACT_BITS(NORMAL_MAP_BITS,       NORMAL_MAP_OFFSET,      value)
#define EXTRACT_OCCLUSION_MAP(value)    EXTRACT_BITS(OCCLUSION_MAP_BITS,    OCCLUSION_MAP_OFFSET,   value)
#define EXTRACT_LIGHT_MAP(value)        EXTRACT_BITS(LIGHT_MAP_BITS,        LIGHT_MAP_OFFSET,       value)  
#define EXTRACT_HEIGHT_MAP(value)       EXTRACT_BITS(HEIGHT_MAP_BITS,       HEIGHT_MAP_OFFSET,      value)
#define EXTRACT_ALPHA_MODE(value)       EXTRACT_BITS(ALPHA_MODE_BITS,       ALPHA_MODE_OFFSET,      value)
#define EXTRACT_DOUBLE_SIDED(value)     EXTRACT_BITS(DOUBLE_SIDED_BITS,     DOUBLE_SIDED_OFFSET,    value)

// Pack bits
#define PACK_BITS(bits, offset, flags, value) (((value & ((1 << bits) - 1)) << offset) | (flags & (~(((1 << bits) - 1) << offset))))
#define PACK_SHADING_MODEL(flags, value)    PACK_BITS(SHADING_MODEL_BITS,    SHADING_MODEL_OFFSET,   flags, value)
#define PACK_DIFFUSE_TYPE(flags, value)     PACK_BITS(DIFFUSE_TYPE_BITS,     DIFFUSE_TYPE_OFFSET,    flags, value)
#define PACK_SPECULAR_TYPE(flags, value)    PACK_BITS(SPECULAR_TYPE_BITS,    SPECULAR_TYPE_OFFSET,   flags, value)
#define PACK_EMISSIVE_TYPE(flags, value)    PACK_BITS(EMISSIVE_TYPE_BITS,    EMISSIVE_TYPE_OFFSET,   flags, value)
#define PACK_NORMAL_MAP_TYPE(flags, value)  PACK_BITS(NORMAL_MAP_BITS,       NORMAL_MAP_OFFSET,      flags, value)
#define PACK_OCCLUSION_MAP(flags, value)    PACK_BITS(OCCLUSION_MAP_BITS,    OCCLUSION_MAP_OFFSET,   flags, value)
#define PACK_LIGHT_MAP(flags, value)        PACK_BITS(LIGHT_MAP_BITS,        LIGHT_MAP_OFFSET,       flags, value)
#define PACK_HEIGHT_MAP(flags, value)       PACK_BITS(HEIGHT_MAP_BITS,       HEIGHT_MAP_OFFSET,      flags, value)
#define PACK_ALPHA_MODE(flags, value)       PACK_BITS(ALPHA_MODE_BITS,       ALPHA_MODE_OFFSET,      flags, value)
#define PACK_DOUBLE_SIDED(flags, value)     PACK_BITS(DOUBLE_SIDED_BITS,     DOUBLE_SIDED_OFFSET,    flags, value)

/*******************************************************************
                    Lights
*******************************************************************/

/**
    Types of light sources. Used in LightData structure.
*/
#define LightPoint                  0    ///< Point light source, can be a spot light if its opening angle is < 2pi
#define LightDirectional            1    ///< Directional light source
#define LightArea                   2    ///< Area light source, potentially with arbitrary geometry
#define LightAreaRect               3    ///< Quad shaped area light source
#define LightAreaSphere             4    ///< Spherical area light source
#define LightAreaDisc               5    ///< Disc shaped area light source

#define MAX_LIGHT_SOURCES 16
#define MAX_AREALIGHT_SOURCES 16

// To bind area lights, use this macro to declare the constant buffer in your shader
#define AREA_LIGHTS shared cbuffer InternalAreaLightCB \
{ \
    int gAreaLightsCount;\
    AreaLightData gAreaLights[MAX_AREALIGHT_SOURCES]; \
};

/** Light probe types
*/
#define LightProbeLinear2D          0    ///< Light probe filtered with linear-filtering, 2D texture
#define LightProbePreIntegrated2D   1    ///< Pre-integrated light probe, 2D texture
#define LightProbeLinearCube        2    ///< Light probe filtered with linear-filtering, texture-cube
#define LightProbePreIntegratedCube 3    ///< Pre-integrated light probe, texture-cube

/*******************************************************************
                Math
*******************************************************************/
#define M_PI     3.14159265358979323846
#define M_PI2    6.28318530717958647692
#define M_INV_PI 0.3183098861837906715

#ifndef HOST_CODE

// Constants from <math.h>
#define M_E                 2.71828182845904523536  // e
#define M_LOG2E             1.44269504088896340736  // log2(e)
#define M_LOG10E            0.434294481903251827651 // log10(e)
#define M_LN2               0.693147180559945309417 // ln(2)
#define M_LN10              2.30258509299404568402  // ln(10)
//#define M_PI                3.14159265358979323846  // pi
#define M_PI_2              1.57079632679489661923  // pi/2
#define M_PI_4              0.785398163397448309616 // pi/4
#define M_1_PI              0.318309886183790671538 // 1/pi
#define M_2_PI              0.636619772367581343076 // 2/pi
#define M_2_SQRTPI          1.12837916709551257390  // 2/sqrt(pi)
#define M_SQRT2             1.41421356237309504880  // sqrt(2)
#define M_SQRT1_2           0.707106781186547524401 // 1/sqrt(2)

// Additional constants
#define M_2PI               6.28318530717958647693  // 2pi
#define M_4PI               12.5663706143591729539  // 4pi
#define M_4_PI              1.27323954473516268615  // 4/pi
#define M_1_2PI             0.159154943091895335769 // 1/2pi
#define M_1_4PI             0.079577471545947667884 // 1/4pi
#define M_SQRTPI            1.77245385090551602730  // sqrt(pi)
#define M_1_SQRT2           0.707106781186547524401 // 1/sqrt(2)

// Numeric limits from <float.h>
#define DBL_DECIMAL_DIG     17                      // # of decimal digits of rounding precision
#define DBL_DIG             15                      // # of decimal digits of precision
#define DBL_EPSILON         2.2204460492503131e-016 // smallest such that 1.0+DBL_EPSILON != 1.0
#define DBL_HAS_SUBNORM     1                       // type does support subnormal numbers
#define DBL_MANT_DIG        53                      // # of bits in mantissa
#define DBL_MAX             1.7976931348623158e+308 // max value
#define DBL_MAX_10_EXP      308                     // max decimal exponent
#define DBL_MAX_EXP         1024                    // max binary exponent
#define DBL_MIN             2.2250738585072014e-308 // min positive value
#define DBL_MIN_10_EXP      (-307)                  // min decimal exponent
#define DBL_MIN_EXP         (-1021)                 // min binary exponent
#define DBL_RADIX           2                       // exponent radix
#define DBL_TRUE_MIN        4.9406564584124654e-324 // min positive value

#define FLT_DECIMAL_DIG     9                       // # of decimal digits of rounding precision
#define FLT_DIG             6                       // # of decimal digits of precision
#define FLT_EPSILON         1.192092896e-07F        // smallest such that 1.0+FLT_EPSILON != 1.0
#define FLT_HAS_SUBNORM     1                       // type does support subnormal numbers
#define FLT_GUARD           0
#define FLT_MANT_DIG        24                      // # of bits in mantissa
#define FLT_MAX             3.402823466e+38F        // max value
#define FLT_MAX_10_EXP      38                      // max decimal exponent
#define FLT_MAX_EXP         128                     // max binary exponent
#define FLT_MIN             1.175494351e-38F        // min normalized positive value
#define FLT_MIN_10_EXP      (-37)                   // min decimal exponent
#define FLT_MIN_EXP         (-125)                  // min binary exponent
#define FLT_NORMALIZE       0
#define FLT_RADIX           2                       // exponent radix
#define FLT_TRUE_MIN        1.401298464e-45F        // min positive value

// Numeric limits for half (IEEE754 binary16)
#define HLF_EPSILON         9.765625e-04F           // smallest such that 1.0+HLF_EPSILON != 1.0
#define HLF_HAS_SUBNORM     1                       // type does support subnormal numbers
#define HLF_MANT_DIG        11
#define HLF_MAX             6.5504e+4F              // max value
#define HLF_MAX_EXP         16                      // max binary exponent
#define HLF_MIN             6.097555160522461e-05F  // min normalized positive value
#define HLF_MIN_EXP         (-14)                   // min binary exponent
#define HLF_RADIX           2
#define HLF_TRUE_MIN        5.960464477539063e-08F  // min positive value
#endif

#endif //_HOST_DEVICE_SHARED_MACROS_H

