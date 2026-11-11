#include "PhysicsRuntimeInternal.h"
#include <cstdlib>

namespace lab {

uint32_t PhysicsRuntime::Impl::Scene::ConfigureSnowfall(uint32_t seed){
    // CreateSnowfall is independent of insect population and its flight seed.
    // Visual publications must not already exist: their resource references
    // are assembled after the weather field is configured.
    if(snowfallReady||!slots[0].refs.empty())return 0;
    PxScopedCudaLock lock(*runtime.cuda);
    constexpr uint64_t bytes=uint64_t(GpuParticleFieldCount)*sizeof(GpuParticleFieldState);
    for(auto& state:snowfallStates)state.Create(owner.interop,bytes);
    Cuda(cudaStreamCreateWithFlags(&snowfallStream,cudaStreamNonBlocking),"create independent snowfall stream");
    const char* windMode=std::getenv("LABOVR_EXPERIMENT_CELL_WIND");
    if(windMode&&windMode[0]=='1'&&windMode[1]=='\0'){
        Cuda(cudaMalloc(reinterpret_cast<void**>(&snowfallWindCells),
            size_t(GpuParticleWindSampleCount)*sizeof(GpuParticleWindSample)),
            "allocate experimental localized snowfall wind");
    }
    snowfallCurrentSlot=0;
    snowfallPendingSeconds=0.;
    snowfallTimeSeconds=0.;
    GpuParticleFieldInitialize(snowfallStream,
        static_cast<GpuParticleFieldState*>(snowfallStates[0].pointer),
        GpuParticleFieldCount,seed);
    Cuda(cudaGetLastError(),"initialize shared CUDA weather snowfall");
    snowfallReady=true;
    LogEvent(0,"physics.particles",
        "count=%u bytes=%llu backend=cuda update_clock=display_frame visible_snow=1 storage=triple_shared state_copy_per_frame=0 independent_of_insects=1 seed=%u cell_wind_approx=%u",
        GpuParticleFieldCount,static_cast<unsigned long long>(bytes),seed,unsigned(snowfallWindCells!=nullptr));
    return 1;
}

void PhysicsRuntime::Impl::Scene::QueueSnowfall(){
    // The dedicated non-blocking stream serializes all snowfall advances and
    // initial state creation. It never reads PhysX poses, forces, or lights.
    auto& publication=*pendingPublication;
    auto& snow=Group(2,publication.groupSlots[2]);
    const uint32_t destinationSlot=publication.groupSlots[2];

    // Wait on the GPU for this destination's last DirectX readers, including
    // Snow.Splat. The source is read-only and prior advances on this stream
    // finish in their original order, even when source==destination.
    snow.consumer.Wait(snowfallStream,publication.groupPriorConsumerValues[2]);
    auto* sample=BeginTiming(particleTimings,particleTimingCursor,tick,0,snowfallStream);
    const auto* source=static_cast<const GpuParticleFieldState*>(snowfallStates[snowfallCurrentSlot].pointer);
    auto* destination=static_cast<GpuParticleFieldState*>(snowfallStates[destinationSlot].pointer);
    if(snowfallWindCells)
        GpuParticleFieldAdvanceCells(snowfallStream,source,destination,GpuParticleFieldCount,
            float(snowfallPendingSeconds),float(snowfallTimeSeconds),snowfallWindCells);
    else
        GpuParticleFieldAdvance(snowfallStream,source,destination,GpuParticleFieldCount,
            float(snowfallPendingSeconds),float(snowfallTimeSeconds));
    Cuda(cudaGetLastError(),"integrate snowfall on dedicated CUDA stream");
    if(sample)Cuda(cudaEventRecord(sample->end,snowfallStream),"end shared snowfall GPU timing");
    snowfallCurrentSlot=destinationSlot;
    snowfallPendingSeconds=0.;
    // CUDA signal is submitted before the renderer can submit its D3D wait.
    snow.producer.Signal(snowfallStream,snow.value);
    snow.submission->SignalEnqueued(snow.value);
}

} // namespace lab
