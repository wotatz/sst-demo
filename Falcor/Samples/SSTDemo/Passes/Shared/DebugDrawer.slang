__import ShaderCommon;
__import DefaultVS;


struct DebugDrawVSIn
{
    float3 position : POSITION;
    float3 color : COLOR;
};

struct DebugDrawVSOut
{
    float4 position : SV_POSITION;
    float3 color : COLOR;
};

DebugDrawVSOut debugDrawVs(DebugDrawVSIn vIn)
{
    DebugDrawVSOut vOut;

    vOut.position =  mul(float4(vIn.position, 1), gCamera.viewProjMat);
    vOut.color = vIn.color;

    return vOut;
}

float4 debugDrawPs(DebugDrawVSOut vOut) : SV_TARGET
{
    return float4(vOut.color, 1);
}
