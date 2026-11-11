#include "GpuParticleField.h"
#include <cmath>
namespace lab {namespace {
__device__ __forceinline__ unsigned Mix(unsigned h){h^=h>>16;h*=0x7feb352du;h^=h>>15;h*=0x846ca68bu;return h^(h>>16);}
__device__ __forceinline__ float Fraction(unsigned h){return float(Mix(h)&0x00ffffffu)*(1.f/16777216.f);}
__global__ void Init(GpuParticleFieldState* p,unsigned count,unsigned seed){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;unsigned key=Mix(i^seed);
    float x=(Fraction(key+11u)-.5f)*256.f,y=.04f+Fraction(key+29u)*17.96f,z=(Fraction(key+47u)-.5f)*256.f;
    float phase=Fraction(key+67u),speed=.65f+1.15f*Fraction(key+79u);
    p[i].position=make_float4(x,y,z,phase);
    p[i].velocity=make_float4((Fraction(key+73u)-.5f)*.35f,-speed,(Fraction(key+97u)-.5f)*.35f,.5f+Fraction(key+103u));
}
__device__ __forceinline__ GpuParticleFieldState Integrate(GpuParticleFieldState s,float dt,float flutter,float swirl){
    s.velocity.x+=(.14f+flutter*.22f*s.velocity.w-s.velocity.x*.20f)*dt;
    s.velocity.z+=(.07f+swirl*.19f*s.velocity.w-s.velocity.z*.20f)*dt;
    float terminal=.65f+1.15f*s.position.w;
    s.velocity.y+=(-terminal-s.velocity.y)*fminf(1.f,dt*2.f);
    s.position.x+=s.velocity.x*dt;s.position.y+=s.velocity.y*dt;s.position.z+=s.velocity.z*dt;
    // Snow settles and is reborn above the street, never bounces like hail.
    if(s.position.y<.04f){s.position.y+=18.f;s.position.x+=(s.position.w-.5f)*3.6f;s.position.z+=(s.velocity.w-.75f)*3.2f;}
    if(s.position.x>=128.f)s.position.x-=256.f;if(s.position.x< -128.f)s.position.x+=256.f;
    if(s.position.z>=128.f)s.position.z-=256.f;if(s.position.z< -128.f)s.position.z+=256.f;
    return s;
}
__global__ void Fall(const GpuParticleFieldState* source,GpuParticleFieldState* target,unsigned count,float dt,float time){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    auto s=source[i];
    // Publication without an intervening simulation frame still receives an
    // identical, complete state; this supports pause and multiple acquisitions.
    if(dt<=0){target[i]=s;return;}float phase=s.position.w*6.2831853f;
    float flutter=__sinf(phase+time*.85f+s.position.z*.035f),swirl=__cosf(phase*.73f+time*.58f+s.position.x*.025f);
    target[i]=Integrate(s,dt,flutter,swirl);
}
// Experimental physical field: 64x64 spatial cells and 32 phase classes.
// Two million independent state records remain; only the wind evaluations
// are approximated at cell centres. This is NOT equivalent to Fall().
__global__ void BuildWindCells(GpuParticleWindSample* samples,float time){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=GpuParticleWindSampleCount)return;
    unsigned phaseBin=i%GpuParticleWindPhaseBins;
    unsigned cellX=(i/GpuParticleWindPhaseBins)%GpuParticleWindCellSide;
    unsigned cellZ=i/(GpuParticleWindPhaseBins*GpuParticleWindCellSide);
    float centerX=-128.f+(float(cellX)+.5f)*(256.f/float(GpuParticleWindCellSide));
    float centerZ=-128.f+(float(cellZ)+.5f)*(256.f/float(GpuParticleWindCellSide));
    float phase=(float(phaseBin)+.5f)*(6.2831853f/float(GpuParticleWindPhaseBins));
    samples[i]={__sinf(phase+time*.85f+centerZ*.035f),
                __cosf(phase*.73f+time*.58f+centerX*.025f)};
}
__device__ __forceinline__ unsigned CellIndex(float x){
    // The original integrator keeps positions in [-128,128), including wrap.
    int index=int(floorf((x+128.f)*(float(GpuParticleWindCellSide)/256.f)));
    return unsigned(index<0?0:(index>=int(GpuParticleWindCellSide)?int(GpuParticleWindCellSide)-1:index));
}
__global__ void FallCells(const GpuParticleFieldState* source,GpuParticleFieldState* target,
                          unsigned count,float dt,const GpuParticleWindSample* samples){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    auto s=source[i];
    if(dt<=0){target[i]=s;return;}
    unsigned phase=unsigned(s.position.w*float(GpuParticleWindPhaseBins));
    if(phase>=GpuParticleWindPhaseBins)phase=GpuParticleWindPhaseBins-1;
    unsigned x=CellIndex(s.position.x),z=CellIndex(s.position.z);
    unsigned cell=(z*GpuParticleWindCellSide+x)*GpuParticleWindPhaseBins+phase;
    auto wind=samples[cell];
    target[i]=Integrate(s,dt,wind.flutter,wind.swirl);
}
}}
namespace lab {
void GpuParticleFieldInitialize(cudaStream_t stream,GpuParticleFieldState* p,std::uint32_t count,std::uint32_t seed){if(count)Init<<<(count+255)/256,256,0,stream>>>(p,count,seed);}
void GpuParticleFieldAdvance(cudaStream_t stream,const GpuParticleFieldState* source,GpuParticleFieldState* target,std::uint32_t count,float dt,float time){if(count)Fall<<<(count+255)/256,256,0,stream>>>(source,target,count,dt,time);}
void GpuParticleFieldAdvanceCells(cudaStream_t stream,const GpuParticleFieldState* source,GpuParticleFieldState* target,std::uint32_t count,float dt,float time,GpuParticleWindSample* cells){
    if(!count)return;
    if(dt>0)BuildWindCells<<<(GpuParticleWindSampleCount+255)/256,256,0,stream>>>(cells,time);
    FallCells<<<(count+255)/256,256,0,stream>>>(source,target,count,dt,cells);
}
}
