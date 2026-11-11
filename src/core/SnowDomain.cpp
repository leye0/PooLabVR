#include "SnowDomain.h"
#include "GpuParticleField.h"
#include "../cartridge_api/LaboVRPopulation.h"
#include "Assets.h"
#include "Core.h"
#include <array>
#include <cstdlib>
#include <vector>
namespace lab {
namespace {
using Microsoft::WRL::ComPtr;
template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE K,class T>struct alignas(void*) Part{D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type=K;T data{};};
struct RasterPipeline {
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,ID3D12RootSignature*> root;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS,D3D12_SHADER_BYTECODE> vs;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,D3D12_SHADER_BYTECODE> ps;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,D3D12_BLEND_DESC> blend;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,UINT> mask;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,D3D12_RASTERIZER_DESC> raster;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,D3D12_DEPTH_STENCIL_DESC> depth;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,D3D12_RT_FORMAT_ARRAY> formats;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,DXGI_SAMPLE_DESC> samples;
 Part<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING,D3D12_VIEW_INSTANCING_DESC> views;
};
}
struct SnowDomain::Impl {
 GpuDomainServices* api{};ComPtr<ID3D12Device2> device;
 ComPtr<ID3D12PipelineState> clear,splat,stereoSplat,composite;
 bool stereoBatch{};
 uint32_t pipelineViews{},width{},height{},viewCount{};
 // Keep coverage and illuminated-colour selection separate: integer atomics
 // are supported on R32_UINT typed UAVs, not on packed multi-channel UAVs.
 std::array<GpuResourceLease,3> accumulation{},illumination{};
 std::array<D3D12_BARRIER_LAYOUT,3> layouts{D3D12_BARRIER_LAYOUT_COMMON,D3D12_BARRIER_LAYOUT_COMMON,D3D12_BARRIER_LAYOUT_COMMON};
 std::array<D3D12_BARRIER_LAYOUT,3> illuminationLayouts{D3D12_BARRIER_LAYOUT_COMMON,D3D12_BARRIER_LAYOUT_COMMON,D3D12_BARRIER_LAYOUT_COMMON};
 struct Source{ComPtr<ID3D12Resource> resource;GpuResourceHandle srv{};uint64_t lastFrame{};};
 std::vector<Source> sources;
 struct Stage{Impl* self{};D3D12_GPU_VIRTUAL_ADDRESS frame{};D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
     uint32_t clearParams[4]{},splatParams[4]{},compositeParams[4]{};
     uint32_t viewCount{},fullWidth{},fullHeight{},slot{};bool recorded{};}stage{};
 ~Impl(){if(api)for(auto& source:sources)api->ReleaseDescriptor(source.srv);}
 void Initialize(GpuDomainServices& services){
     api=&services;Hr(api->Device()->QueryInterface(IID_PPV_ARGS(&device)),"Snowfall device2");
     auto make=[&](const wchar_t* entry,ComPtr<ID3D12PipelineState>& pipeline){
         auto bytes=ReadBytes(AssetPath((std::wstring(L"shaders/")+entry+L".cso").c_str()));
         D3D12_COMPUTE_PIPELINE_STATE_DESC d{};d.pRootSignature=api->BindlessRootSignature();d.CS={bytes.data(),bytes.size()};
         Hr(device->CreateComputePipelineState(&d,IID_PPV_ARGS(&pipeline)),"Snowfall compute pipeline");
     };
     make(L"SnowClear",clear);make(L"SnowSplat",splat);
     make(L"SnowSplatStereo",stereoSplat);sources.reserve(4);
     // Opt-in until GPU/VR diagnostics can verify occupancy and frame latency.
     const char* selection=std::getenv("LABOVR_EXPERIMENT_STEREO_SNOW");
     stereoBatch=selection&&selection[0]=='1'&&selection[1]=='\0';
 }
 void Pipeline(uint32_t views){
     if(pipelineViews==views)return;if(!views||views>4)Fail("Snowfall invalid view topology");
     std::array<D3D12_VIEW_INSTANCE_LOCATION,4> locations{};for(uint32_t i=0;i<views;i++)locations[i]={0,i};
     auto vs=ReadBytes(AssetPath(L"shaders/SnowCompositeVS.cso")),ps=ReadBytes(AssetPath(L"shaders/SnowCompositePS.cso"));
     RasterPipeline p{};p.root.data=api->BindlessRootSignature();p.vs.data={vs.data(),vs.size()};p.ps.data={ps.data(),ps.size()};
     p.mask.data=UINT_MAX;p.raster.data.FillMode=D3D12_FILL_MODE_SOLID;
     p.raster.data.CullMode=D3D12_CULL_MODE_NONE;p.raster.data.DepthClipEnable=TRUE;
     p.depth.data.DepthEnable=FALSE;p.depth.data.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO;
     p.depth.data.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
     p.topology.data=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
     p.formats.data.NumRenderTargets=1;p.formats.data.RTFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT;
     p.samples.data={1,0};p.views.data={views,locations.data(),D3D12_VIEW_INSTANCING_FLAG_NONE};
     auto& blend=p.blend.data.RenderTarget[0];blend.BlendEnable=TRUE;
     blend.SrcBlend=D3D12_BLEND_SRC_ALPHA;blend.DestBlend=D3D12_BLEND_INV_SRC_ALPHA;
     blend.BlendOp=D3D12_BLEND_OP_ADD;blend.SrcBlendAlpha=D3D12_BLEND_ZERO;
     blend.DestBlendAlpha=D3D12_BLEND_ONE;blend.BlendOpAlpha=D3D12_BLEND_OP_ADD;
     blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
     D3D12_PIPELINE_STATE_STREAM_DESC d{sizeof(p),&p};
     Hr(device->CreatePipelineState(&d,IID_PPV_ARGS(&composite)),"Snowfall stereo pipeline");
     pipelineViews=views;
 }
 uint32_t SourceSrv(const GpuPublishedResource& resource,uint64_t frame){
     for(auto& s:sources)if(s.resource.Get()==resource.resource.Get()){s.lastFrame=frame;return api->DescriptorIndex(s.srv);}
     if(resource.bytes!=uint64_t(GpuParticleFieldCount)*sizeof(GpuParticleFieldState))Fail("Snowfall shared buffer size");
     Source s{};s.resource=resource.resource;
     s.srv=api->CreateBufferSrv(resource.resource.Get(),GpuParticleFieldCount,sizeof(GpuParticleFieldState));
     if(!s.srv.Valid())Fail("Snowfall source descriptor");
     s.lastFrame=frame;auto index=api->DescriptorIndex(s.srv);sources.push_back(std::move(s));return index;
 }
 void Allocate(const GpuFrameContext& ctx){
     uint32_t w=(ctx.width+1)/2,h=(ctx.height+1)/2;
     if(width==w&&height==h&&viewCount==ctx.viewCount&&accumulation[0]&&illumination[0])return;
     for(auto& t:accumulation)t.reset();
     for(auto& t:illumination)t.reset();
     layouts.fill(D3D12_BARRIER_LAYOUT_COMMON);
     illuminationLayouts.fill(D3D12_BARRIER_LAYOUT_COMMON);
     GpuTextureDesc d{};d.width=w;d.height=h;d.depthOrArraySize=ctx.viewCount;
     d.format=DXGI_FORMAT_R32_UINT;d.name=L"Snowfall.Coverage";
     for(auto& t:accumulation){t=api->CreateManagedTexture2D(d);if(!t)Fail("Snowfall coverage allocation");}
     d.name=L"Snowfall.LampIllumination";
     for(auto& t:illumination){t=api->CreateManagedTexture2D(d);if(!t)Fail("Snowfall illumination allocation");}
     width=w;height=h;viewCount=ctx.viewCount;
 }
 void Bind(ID3D12GraphicsCommandList7* l,bool compute,const uint32_t constants[4]){
     ID3D12DescriptorHeap* heaps[]{api->BindlessResourceHeap(),api->BindlessSamplerHeap()};
     l->SetDescriptorHeaps(2,heaps);
     if(compute){l->SetComputeRootSignature(api->BindlessRootSignature());
         l->SetComputeRootConstantBufferView(0,stage.frame);l->SetComputeRoot32BitConstants(1,4,constants,0);}
     else{l->SetGraphicsRootSignature(api->BindlessRootSignature());
         l->SetGraphicsRootConstantBufferView(0,stage.frame);l->SetGraphicsRoot32BitConstants(1,4,constants,0);}
 }
 static void Clear(void* ptr,ID3D12GraphicsCommandList7* l){
     auto& s=*static_cast<Stage*>(ptr);s.self->Bind(l,true,s.clearParams);
     l->SetPipelineState(s.self->clear.Get());l->Dispatch((s.self->width+7)/8,(s.self->height+7)/8,s.viewCount);
 }
 static void Splat(void* ptr,ID3D12GraphicsCommandList7* l){
     auto& s=*static_cast<Stage*>(ptr);s.self->Bind(l,true,s.splatParams);
     l->SetPipelineState(s.self->stereoBatch?s.self->stereoSplat.Get():s.self->splat.Get());
     l->Dispatch((GpuParticleFieldCount+255)/256,s.self->stereoBatch?1:s.viewCount,1);
 }
 static void Composite(void* ptr,ID3D12GraphicsCommandList7* l){
     auto& s=*static_cast<Stage*>(ptr);s.self->Bind(l,false,s.compositeParams);
     l->SetPipelineState(s.self->composite.Get());
     D3D12_VIEWPORT vp{0,0,float(s.fullWidth),float(s.fullHeight),0,1};
     D3D12_RECT rect{0,0,LONG(s.fullWidth),LONG(s.fullHeight)};
     l->RSSetViewports(1,&vp);l->RSSetScissorRects(1,&rect);
     l->OMSetRenderTargets(1,&s.rtv,FALSE,nullptr);
     l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
     l->DrawInstanced(3,1,0,0);s.recorded=true;
 }
 void Build(GpuPassBuilder& b,const GpuFrameContext& ctx){
     if(!api||!ctx.resources||!ctx.viewCount)return;
     const GpuPublishedResource* particles{};
     for(uint32_t i=0;i<64;i++){
         auto* source=ctx.FindPublication(GpuDomainId::MeshSolid,i);if(!source)break;
         particles=source->Find(LAB_SNOW_FIELD_BINDING_V1);
         if(particles&&!(particles->flags&1))break;
         particles=nullptr;
     }
     if(!particles)return;
     Pipeline(ctx.viewCount);Allocate(ctx);
     const auto& color=ctx.Require(GpuFrameResourceId::ResolvedColor);
     const auto& depth=ctx.Require(GpuFrameResourceId::SceneDepth);
     const auto& lights=ctx.Require(GpuFrameResourceId::Lighting);
     const auto& clusters=ctx.Require(GpuFrameResourceId::Clusters);
     uint32_t slot=ctx.uploadSlot;if(slot>=accumulation.size())Fail("Snowfall frame slot");
     const auto& cover=accumulation[slot];
     const auto& lit=illumination[slot];
     stage={};stage.self=this;stage.frame=ctx.frameConstants;stage.rtv=color.rtv;
     stage.slot=slot;stage.viewCount=ctx.viewCount;stage.fullWidth=ctx.width;stage.fullHeight=ctx.height;
     stage.clearParams[0]=api->DescriptorIndex(cover->Uav());
     stage.clearParams[1]=api->DescriptorIndex(lit->Uav());
     stage.splatParams[0]=SourceSrv(*particles,ctx.frameId);
     stage.splatParams[1]=api->DescriptorIndex(cover->Uav());
     stage.splatParams[2]=api->DescriptorIndex(depth.srv);
     stage.splatParams[3]=api->DescriptorIndex(lit->Uav());
     stage.compositeParams[0]=api->DescriptorIndex(cover->srv);
     stage.compositeParams[1]=api->DescriptorIndex(lit->srv);
     b.BeginPass("Snow.Clear",GpuPassQueue::Compute,{&stage,&Clear});
     b.Texture(cover->resource,GpuPassResourceAccess::Write,layouts[slot],D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
               D3D12_BARRIER_SYNC_COMPUTE_SHADING,D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
     b.Texture(lit->resource,GpuPassResourceAccess::Write,illuminationLayouts[slot],D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
               D3D12_BARRIER_SYNC_COMPUTE_SHADING,D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);b.EndPass();
     b.BeginPass("Snow.Splat",GpuPassQueue::Compute,{&stage,&Splat});
     b.Buffer(particles->resource.Get(),GpuPassResourceAccess::Read,D3D12_BARRIER_SYNC_COMPUTE_SHADING,D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
     b.Texture(depth.resource,GpuPassResourceAccess::Read,depth.initialLayout,D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
               D3D12_BARRIER_SYNC_COMPUTE_SHADING,D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
     // Reuse the scene's live DX12 light buffer and clustered-light culling:
     // no duplicate lamps, no CPU light upload, and no per-flake ray query.
     b.Buffer(lights.resource,GpuPassResourceAccess::Read,D3D12_BARRIER_SYNC_COMPUTE_SHADING,
              D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
     b.Buffer(clusters.resource,GpuPassResourceAccess::Read,D3D12_BARRIER_SYNC_COMPUTE_SHADING,
              D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
     b.Texture(cover->resource,GpuPassResourceAccess::ReadWrite,D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
               D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,D3D12_BARRIER_SYNC_COMPUTE_SHADING,
               D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);
     b.Texture(lit->resource,GpuPassResourceAccess::ReadWrite,D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
               D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,D3D12_BARRIER_SYNC_COMPUTE_SHADING,
               D3D12_BARRIER_ACCESS_UNORDERED_ACCESS);b.EndPass();
     b.BeginPass("Snow.Composite",GpuPassQueue::Direct,{&stage,&Composite});
     b.Texture(cover->resource,GpuPassResourceAccess::Read,D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
               D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,D3D12_BARRIER_SYNC_PIXEL_SHADING,D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
     b.Texture(lit->resource,GpuPassResourceAccess::Read,D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
               D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,D3D12_BARRIER_SYNC_PIXEL_SHADING,D3D12_BARRIER_ACCESS_SHADER_RESOURCE);
     b.Texture(color.resource,GpuPassResourceAccess::ReadWrite,color.initialLayout,D3D12_BARRIER_LAYOUT_RENDER_TARGET,
               D3D12_BARRIER_SYNC_RENDER_TARGET,D3D12_BARRIER_ACCESS_RENDER_TARGET);b.EndPass();
     for(auto it=sources.begin();it!=sources.end();){
         if(ctx.frameId>it->lastFrame+12){api->ReleaseDescriptor(it->srv);it=sources.erase(it);}
         else ++it;
     }
 }
 void End(){if(stage.recorded){layouts[stage.slot]=D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;illuminationLayouts[stage.slot]=D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;}stage={};}
};
SnowDomain::SnowDomain():impl_(std::make_unique<Impl>()){}
SnowDomain::~SnowDomain()=default;
void SnowDomain::Initialize(GpuDomainServices& s){impl_->Initialize(s);}
void SnowDomain::BuildPasses(GpuPassBuilder& b,const GpuFrameContext& c){impl_->Build(b,c);}
void SnowDomain::EndFrame(){impl_->End();}
}
