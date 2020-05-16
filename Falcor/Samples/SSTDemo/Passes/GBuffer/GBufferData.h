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
#include "Falcor.h"

// GBuffer channel metadata
struct GBufferChannelDesc
{
  const char*    name;           // Canonical channel name
  const char*    desc;           // Human-readable channel description
  const char*    texname;        // Name of corresponding ITexture2D in GBufferRT shader code
  Falcor::ResourceFormat format; // Resource format
};

static const std::vector<GBufferChannelDesc> kGBufferChannelDesc({
        {"PosW",             "world space position",                                               "gPosW",             Falcor::ResourceFormat::RGBA32Float },
        {"Packed1",          "world space normal, emissive color",                                 "gPacked1",          Falcor::ResourceFormat::RGBA32Uint },
        {"Packed2",          "material diffuse, opacity, material specular, roughness",            "gPacked2",          Falcor::ResourceFormat::RGBA32Uint },
        {"CNNAux",           "CNN Auxiliary (normV.xy, depth, roughness)",                         "gCNNAux",           Falcor::ResourceFormat::RGBA32Float },
        {"Motion",           "motion vector and fwidth of pos & normal",                           "gMotion",           Falcor::ResourceFormat::RG32Float },
        {"PosNormalFwidth",  "derivatives of position and normal",                                 "gPosNormalFwidth",  Falcor::ResourceFormat::RG32Float },
        {"LinearZAndNormal", "linear z and its derivative + normal",                               "gLinearZAndNormal", Falcor::ResourceFormat::RGBA32Float },
  });
