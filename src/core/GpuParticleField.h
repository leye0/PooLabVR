#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace lab {
// Independently simulated GPU micro-particles. They do not replace PhysX rigid
// bodies, the existing 500 flies, or the atmospheric rendering pipeline.
inline constexpr std::uint32_t GpuParticleFieldCount=2'000'000u;
inline constexpr std::uint32_t GpuParticleWindCellSide=64u;
inline constexpr std::uint32_t GpuParticleWindPhaseBins=32u;
inline constexpr std::uint32_t GpuParticleWindSampleCount=GpuParticleWindCellSide*GpuParticleWindCellSide*GpuParticleWindPhaseBins;
struct GpuParticleWindSample {float flutter,swirl;};
struct alignas(16) GpuParticleFieldState {
    float4 position;  // xyz in metres, w = deterministic phase
    float4 velocity;  // xyz in metres/second, w = per-particle wind factor
};
static_assert(sizeof(GpuParticleFieldState)==32);
void GpuParticleFieldInitialize(cudaStream_t,GpuParticleFieldState*,std::uint32_t,std::uint32_t);
void GpuParticleFieldAdvance(cudaStream_t,const GpuParticleFieldState*,GpuParticleFieldState*,std::uint32_t,float,float);
// Approximate, opt-in localized wind. Preserves all particle identities/state
// but intentionally does not preserve their exact original trajectories.
void GpuParticleFieldAdvanceCells(cudaStream_t,const GpuParticleFieldState*,GpuParticleFieldState*,std::uint32_t,float,float,GpuParticleWindSample*);
}
