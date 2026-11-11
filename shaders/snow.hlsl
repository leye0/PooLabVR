#include "common.hlsli"
// The 2,000,000 CUDA states are published directly to the renderer. Lighting
// is evaluated only after distance, frustum and scene-depth rejection.
static const uint SnowStateCount=2000000u;
struct SnowParticle {float4 position,velocity;};

[numthreads(8,8,1)]
void SnowClear(uint3 id:SV_DispatchThreadID){
    RWTexture2DArray<uint> coverage=ResourceDescriptorHeap[objectIndex];
    RWTexture2DArray<uint> illuminated=ResourceDescriptorHeap[passIndex];
    uint w,h,views;coverage.GetDimensions(w,h,views);
    if(id.x<w&&id.y<h&&id.z<views){coverage[id]=0;illuminated[id]=0;}
}
float SceneSnowDepth(uint2 pixel,uint view){
    if(topology.y>1){
        Texture2DMSArray<float,4> depth=ResourceDescriptorHeap[param0];float d=0;
        [unroll]for(uint sample=0;sample<4;sample++)d=max(d,depth.Load(int3(pixel,view),sample));return d;
    }
    Texture2DArray<float> depth=ResourceDescriptorHeap[param0];return depth.Load(int4(pixel,view,0));
}

// The same finite-range, squared-edge, inverse-square and spot-cone
// attenuation as the scene and volumetric-fog lighting paths. Actual GPU
// light positions/colours/powers drive the visible flakes, including lights
// switching off. No shadow rays are launched per flake.
float3 SnowLamp(float3 p,Light light){
    if(light.colorPower.w<=0||((uint)light.parameters.z&2u)!=0u)return 0;
    float3 delta=light.positionRadius.xyz-p;
    float distanceSq=dot(delta,delta);
    float radius=light.positionRadius.w;
    if(distanceSq>=radius*radius)return 0;
    float distance=sqrt(distanceSq);
    float3 dir=delta/max(distance,.001);
    float ratioSq=distanceSq/(radius*radius);
    float edge=saturate(1-ratioSq*ratioSq);
    float attenuation=edge*edge/max(.035,distanceSq);
    if(light.directionOuter.w>-1)
        attenuation*=saturate((dot(-dir,light.directionOuter.xyz)-light.directionOuter.w)/
                              max(.00001,light.parameters.x-light.directionOuter.w));
    return light.colorPower.rgb*(light.colorPower.w*attenuation);
}
float3 SnowIncident(float3 p,uint2 pixel,float depth,uint view){
    if(LightCount==0)return 0;
    StructuredBuffer<Light> lights=ResourceDescriptorHeap[Lights];
    float3 incident=0;
    if(LightCount<=4){
        [loop]for(uint i=0;i<LightCount;i++)incident+=SnowLamp(p,lights[i]);
    }else{
        // Renderer clusters reject lamps outside this 64x64/depth region.
        // At most the local bit-mask entries are evaluated per visible flake.
        StructuredBuffer<uint4> clusters=ResourceDescriptorHeap[Clusters];
        uint4 masks=clusters[ClusterIndex(pixel,depth,view)];
        [unroll]for(uint word=0;word<4;word++){
            uint mask=masks[word];
            while(mask){
                uint index=word*32+firstbitlow(mask);mask&=mask-1;
                if(index<LightCount)incident+=SnowLamp(p,lights[index]);
            }
        }
    }
    return incident;
}
// Two R32_UINT UAVs preserve supported typed atomics. Coverage sums as before;
// illuminated selects the strongest light at a half-res pixel, avoiding
// unbounded per-pixel accumulation and packed-channel carry corruption.
// High byte sorts by brightness; lower bytes retain the actual light hue.
uint PackSnowLight(float3 light){
    float peak=max(light.r,max(light.g,light.b));
    if(peak<=.015f)return 0u;
    uint brightness=(uint)(255.f*saturate(peak/(1.f+peak)));
    uint3 hue=(uint3)(255.f*saturate(light/max(peak,.00001f)));
    return (brightness<<24)|(hue.r<<16)|(hue.g<<8)|hue.b;
}
// Shared by the original per-eye dispatch and the experimental one-thread-per-
// particle stereo dispatch. The latter reuses a single particle-state load;
// projection and scene-depth rejection remain exact and per eye.
void SnowSplatEye(SnowParticle s,uint view){
    float3 p=s.position.xyz;
    p.x+=floor((eye[view].x-p.x+128.f)/256.f)*256.f;
    p.z+=floor((eye[view].z-p.z+128.f)/256.f)*256.f;
    // The laboratory is roofed, but the sky and street remain snowy.
    if(abs(p.x)<4.2f&&abs(p.z)<4.2f&&p.y<4.05f)return;
    float3 delta=p-eye[view].xyz;
    if(abs(delta.x)>35.f||abs(delta.z)>35.f||dot(delta,delta)>1225.f)return;
    float4 clip=mul(float4(p,1),vp[view]);
    if(clip.w<=.05f||clip.z<0||clip.z>clip.w)return;
    float2 uv=clip.xy/clip.w*float2(.5f,-.5f)+.5f;
    if(any(uv<=0)||any(uv>=1))return;
    uint2 depthPixel=min(uint2(uv*float2(Width,Height)),uint2(Width-1,Height-1));
    // Reverse-Z: surfaces closer to the eye occlude the snow.
    float depth=clip.z/clip.w;
    if(SceneSnowDepth(depthPixel,view)>depth+.00025f)return;
    RWTexture2DArray<uint> coverage=ResourceDescriptorHeap[passIndex];
    uint w,h,views;coverage.GetDimensions(w,h,views);
    int2 pixel=int2(uv*float2(w,h));if(any(pixel<0)||any(pixel>=int2(w,h)))return;
    float distance=length(delta);
    uint strength=max(1u,(uint)(330.f*saturate((35.f-distance)/15.f)*lerp(.4f,1.f,s.position.w)));
    InterlockedAdd(coverage[uint3(pixel,view)],strength);
    if(distance<9.f&&pixel.x+1<int(w))
        InterlockedAdd(coverage[uint3(pixel+int2(1,0),view)],strength/3u);
    // Lamp illumination is computed only for a surviving visible flake,
    // never for all two million invisible/out-of-range states.
    uint lit=PackSnowLight(SnowIncident(p,depthPixel,depth,view));
    if(lit!=0u){
        RWTexture2DArray<uint> illuminated=ResourceDescriptorHeap[param1];
        InterlockedMax(illuminated[uint3(pixel,view)],lit);
        if(distance<9.f&&pixel.x+1<int(w))
            InterlockedMax(illuminated[uint3(pixel+int2(1,0),view)],lit);
    }
}
[numthreads(256,1,1)]
void SnowSplat(uint3 id:SV_DispatchThreadID){
    uint index=id.x,view=id.y;if(index>=SnowStateCount||view>=ViewCount)return;
    StructuredBuffer<SnowParticle> states=ResourceDescriptorHeap[objectIndex];
    SnowSplatEye(states[index],view);
}
[numthreads(256,1,1)]
void SnowSplatStereo(uint3 id:SV_DispatchThreadID){
    uint index=id.x;if(index>=SnowStateCount)return;
    StructuredBuffer<SnowParticle> states=ResourceDescriptorHeap[objectIndex];
    SnowParticle particle=states[index];
    [loop]for(uint view=0;view<ViewCount;view++)SnowSplatEye(particle,view);
}
struct SnowOut{float4 position:SV_Position;float2 uv:TEXCOORD0;nointerpolation uint view:TEXCOORD1;};
SnowOut SnowCompositeVS(uint index:SV_VertexID,uint view:SV_ViewID){
    SnowOut o;o.uv=float2((index<<1)&2,index&2);
    o.position=float4(o.uv*float2(2,-2)+float2(-1,1),0,1);o.view=view;return o;
}
float3 SnowColour(uint packed){
    const float3 cold=float3(.67f,.76f,.91f);
    if(packed==0u)return cold;
    float illumination=float((packed>>24)&255u)*(1.f/255.f);
    float3 hue=float3((packed>>16)&255u,(packed>>8)&255u,packed&255u)*(1.f/255.f);
    // White snow reflects the lamp's hue; unlit flakes retain exactly the
    // original cool colour. Limit tint to retain natural night contrast.
    return lerp(cold,1.15f*hue,illumination*.85f);
}
float4 SnowCompositePS(SnowOut i):SV_Target0{
    Texture2DArray<uint> coverage=ResourceDescriptorHeap[objectIndex];
    Texture2DArray<uint> lit=ResourceDescriptorHeap[passIndex];
    uint w,h,views;coverage.GetDimensions(w,h,views);
    float2 p=i.uv*float2(w,h)-.5f;int2 base=int2(floor(p));float2 f=frac(p);
    float coverageLevel=0;
    float3 colour=0;
    [unroll]for(int y=0;y<2;y++)[unroll]for(int x=0;x<2;x++){
        int2 q=clamp(base+int2(x,y),int2(0,0),int2(int(w)-1,int(h)-1));
        float weight=(x?f.x:1.f-f.x)*(y?f.y:1.f-f.y);
        coverageLevel+=weight*float(coverage.Load(int4(q,i.view,0)));
        colour+=weight*SnowColour(lit.Load(int4(q,i.view,0)));
    }
    return float4(colour,saturate(coverageLevel/850.f));
}
