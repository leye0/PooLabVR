#include "PhysicsRuntimeInternal.h"
#include "MeshSolidPublication.h"

namespace lab {
void PhysicsRuntime::Impl::Scene::Initialize(){
    PxSceneDesc d(runtime.physics->getTolerancesScale());d.gravity=V(desc.gravity);d.cpuDispatcher=runtime.dispatcher;d.cudaContextManager=runtime.cuda;d.filterShader=Filter;
    d.flags|=PxSceneFlag::eENABLE_GPU_DYNAMICS|PxSceneFlag::eENABLE_DIRECT_GPU_API|PxSceneFlag::eDISABLE_SLEEPING|PxSceneFlag::eENABLE_PCM;
    d.broadPhaseType=PxBroadPhaseType::eGPU;d.solverType=PxSolverType::eTGS;
    // Preallocation scales with the declared scene limit; overflows are errors,
    // never permission to drop contacts silently or substitute a CPU scene.
    d.gpuDynamicsConfig.foundLostPairsCapacity=std::max(1024u,desc.maxBodies*16);
    d.gpuDynamicsConfig.foundLostAggregatePairsCapacity=1024;
    d.gpuDynamicsConfig.totalAggregatePairsCapacity=1024;
    d.gpuDynamicsConfig.heapCapacity=16*1024*1024;d.gpuDynamicsConfig.tempBufferCapacity=4*1024*1024;
    if(!d.isValid())Fail("PhysX GPU scene descriptor is invalid");
    scene=runtime.physics->createScene(d);runtime.Check();
    if(!scene||!scene->getFlags().isSet(PxSceneFlag::eENABLE_GPU_DYNAMICS)||!scene->getFlags().isSet(PxSceneFlag::eENABLE_DIRECT_GPU_API)||!scene->getFlags().isSet(PxSceneFlag::eDISABLE_SLEEPING)||!scene->getFlags().isSet(PxSceneFlag::eENABLE_PCM)||scene->getBroadPhaseType()!=PxBroadPhaseType::eGPU||scene->getSolverType()!=PxSolverType::eTGS||scene->getCudaContextManager()!=runtime.cuda)Fail("PhysX: scene did not retain the required CUDA GPU/TGS contract");
    PxScopedCudaLock lock(*runtime.cuda);Cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create physics command stream");
    for(auto* e:{&uploadReady,&poseReady,&velocityReady,&stateReady,&forceReady,&forceDone,&torqueDone,&contactReady})Driver(cuEventCreate(e,CU_EVENT_DISABLE_TIMING),"create direct GPU API event");
    Allocate(indices,desc.maxBodies);Allocate(previous,desc.maxBodies);Allocate(current,desc.maxBodies);Allocate(velocity,desc.maxBodies);Allocate(angular,desc.maxBodies);Allocate(forces,desc.maxBodies);Allocate(torques,desc.maxBodies);
    Allocate(editPoseDevice,desc.maxBodies);Allocate(editIndexDevice,desc.maxBodies);Allocate(editZeroDevice,desc.maxBodies);
    Cuda(cudaMemsetAsync(editZeroDevice,0,desc.maxBodies*sizeof(PxVec3),stream),"initialize editor zero motion");
    Cuda(cudaHostAlloc(reinterpret_cast<void**>(&editPoseHost),desc.maxBodies*sizeof(PxTransform),cudaHostAllocDefault),"allocate editor pose staging");
    Cuda(cudaHostAlloc(reinterpret_cast<void**>(&editIndexHost),desc.maxBodies*sizeof(PxRigidDynamicGPUIndex),cudaHostAllocDefault),"allocate editor index staging");
    Allocate(airflowDevice,8);Allocate(dragDevice,desc.maxBodies);
    Allocate(visualDevice,MaxVisuals);Allocate(lightDevice,MaxLights);Allocate(aeroDevice,desc.maxBodies);Allocate(probeDevice,LABOVR_PHYSICS_MAX_PROBES_V1);Allocate(actorDevice,desc.maxBodies);
    Allocate(populationBodies,desc.maxBodies);Allocate(populationSummary,1);
    for(auto& r:populationReadbacks){Cuda(cudaHostAlloc(reinterpret_cast<void**>(&r.host),sizeof(LabPopulationSnapshotV1),cudaHostAllocDefault),"allocate compact population readback");Cuda(cudaEventCreateWithFlags(&r.done,cudaEventDisableTiming),"population readback event");}
    Allocate(contacts,ContactCapacity);Allocate(contactCount,1);Allocate(probeOutput,LABOVR_PHYSICS_MAX_PROBES_V1);
    streamMaterials.reserve(MaxVisuals);bodies.reserve(desc.maxBodies);joints.reserve(desc.maxBodies*2);commands.reserve(desc.maxBodies*4);indexHost.reserve(desc.maxBodies);actorHost.reserve(desc.maxBodies);
    // Pinned command arrays avoid the staging/synchronization that pageable
    // host memory can introduce into every fixed step's async uploads. The
    // torqueDone dependency retires both uploads before these arrays are reused.
    Cuda(cudaHostAlloc(reinterpret_cast<void**>(&forceHost),desc.maxBodies*sizeof(PxVec3),cudaHostAllocDefault),"allocate pinned batched forces");
    Cuda(cudaHostAlloc(reinterpret_cast<void**>(&torqueHost),desc.maxBodies*sizeof(PxVec3),cudaHostAllocDefault),"allocate pinned batched torques");
    std::uninitialized_fill_n(forceHost,desc.maxBodies,PxVec3(0));std::uninitialized_fill_n(torqueHost,desc.maxBodies,PxVec3(0));
    for(auto& p:probeSlots){Cuda(cudaHostAlloc(reinterpret_cast<void**>(&p.host),sizeof(LabPhysicsStateV1)*LABOVR_PHYSICS_MAX_PROBES_V1,cudaHostAllocDefault),"allocate asynchronous probe readback");Cuda(cudaHostAlloc(reinterpret_cast<void**>(&p.overflow),sizeof(uint32_t),cudaHostAllocDefault),"allocate contact count readback");Cuda(cudaEventCreateWithFlags(&p.done,cudaEventDisableTiming),"create probe completion");Cuda(cudaEventCreate(&p.gpuBegin),"create GPU timing start");Cuda(cudaEventCreate(&p.gpuEnd),"create GPU timing end");}
    for(auto pool:{std::span<Timing>(timings),std::span<Timing>(stepTimings),std::span<Timing>(publicationTimings),std::span<Timing>(particleTimings)})for(auto& t:pool){Cuda(cudaEventCreate(&t.begin),"create physics GPU span start");Cuda(cudaEventCreate(&t.end),"create physics GPU span end");}
    provider={LABOVR_GPU_ABI_V1,sizeof(provider),this,
        [](void* c,LabGpuPublicationV1* out)->LabU32{return c?static_cast<Scene*>(c)->Acquire(out):0;},
        [](void* c,void* token,LabU64 generation){if(c)static_cast<Scene*>(c)->Release(token,generation);},LABOVR_GPU_DOMAIN_MESH_SOLID_V1,0};
    provider.reserved=LABOVR_GPU_PROVIDER_REBASED_MATERIALS_V1;
    LogEvent(0,"physics.scene","scene=%llu gpu_dynamics=1 gpu_broadphase=1 direct_gpu_api=1 solver=TGS fixed_dt=%.8f capacity=%u",static_cast<unsigned long long>(handle),desc.fixedStepSeconds,desc.maxBodies);
}
void PhysicsRuntime::Impl::Scene::DestroyPublications(){
    if(pendingPublication)Fail("PhysX: destroying an unfinished publication");
    for(auto& s:slots){
        if(s.leased)Fail("PhysX: destroying a publication still acquired by its client");
        if(s.value)s.consumer.Wait(stream,s.value);
        for(auto* g:{&s.physicsVisual,&s.population,&s.snow}){
            if(g->leased)Fail("PhysX: destroying a GPU group still acquired by its client");
            if(g->value)g->consumer.Wait(stream,g->value);
        }
    }
    Cuda(cudaStreamSynchronize(stream),"retire scene publications");
    // The population producer and GPU-only combined-buffer merge must both
    // retire before their shared resources and group fences can be destroyed.
    if(populationStream)Cuda(cudaStreamSynchronize(populationStream),"retire independent population publications");
    // Scene rebuild/destruction cannot retire a shared snow state or fence
    // while its independent CUDA stream still has work in flight.
    if(snowfallStream)Cuda(cudaStreamSynchronize(snowfallStream),"retire independent snowfall publications");
    for(auto& s:slots){
        for(auto* g:{&s.physicsVisual,&s.population,&s.snow}){
            g->instances.Destroy(owner.interop);g->control.Destroy(owner.interop);g->lighting.Destroy(owner.interop);
            g->producer.Destroy(owner.interop);g->consumer.Destroy(owner.interop);
            g->bindings.clear();g->value=g->leaseGeneration=g->storageGeneration=0;g->leased=false;g->submission.reset();
        }
        s.instances.Destroy(owner.interop);s.control.Destroy(owner.interop);
        s.producer.Destroy(owner.interop);s.consumer.Destroy(owner.interop);
        s.refs.clear();s.value=s.leaseGeneration=s.rayRevision=0;s.physicalCount=0;s.groupActive.fill(false);s.submission.reset();
    }
    groupCursors.fill(0);populationCopied.fill(false);
    // Retained descriptor signature is rebuilt on the next descriptor upload.
    // A new source allocation always receives its own storage generation.
}
void PhysicsRuntime::Impl::Scene::Destroy(){
    if(SessionLogCompact()&&diagnostic.frames){
        LogPhysicsProfileCpu(diagnostic,desc.fixedStepSeconds,desc.maxSubsteps,droppedSeconds);
        const uint32_t activeJoints=uint32_t(std::count_if(joints.begin(),joints.end(),
            [](const Joint& j){return j.joint!=nullptr;}));
        LogPhysicsProfileSolver(diagnostic,SolverPositionIterations,SolverVelocityIterations,
                                dynamicCount,activeJoints,scene->getFlags().isSet(PxSceneFlag::eDISABLE_SLEEPING));
    }
    {PxScopedCudaLock lock(*runtime.cuda);DestroyPublications();Cuda(cudaStreamSynchronize(stream),"retire physics scene");
    cudaFreeHost(forceHost);cudaFreeHost(torqueHost);cudaFreeHost(editPoseHost);cudaFreeHost(editIndexHost);cudaFree(editPoseDevice);cudaFree(editIndexDevice);cudaFree(editZeroDevice);
    if(insects)cudaFree(insects);for(auto& snow:snowfallStates)if(snow.handle)snow.Destroy(owner.interop);if(snowfallWindCells)cudaFree(snowfallWindCells);cudaFree(populationBodies);if(populationBodyPoses)cudaFree(populationBodyPoses);if(populationGrid)cudaFree(populationGrid);cudaFree(populationSummary);for(auto& r:populationReadbacks){cudaFreeHost(r.host);cudaEventDestroy(r.done);}
    for(auto& p:probeSlots){cudaFreeHost(p.host);cudaFreeHost(p.overflow);cudaEventDestroy(p.done);cudaEventDestroy(p.gpuBegin);cudaEventDestroy(p.gpuEnd);}
    CollectTimings();
    if(SessionLogCompact()&&diagnostic.frames)LogPhysicsProfileGpu(diagnostic);
    if(snowfallReady&&SessionLogCompact())LogEvent(0,"physics.profile.particles","count=%u gpu_samples=%llu gpu_step_avg_ms=%.4f gpu_step_peak_ms=%.4f state_bytes=%llu storage=triple_shared state_copy_per_frame=0",GpuParticleFieldCount,static_cast<unsigned long long>(particleSamples),particleSamples?particleSumMs/double(particleSamples):0.,particlePeakMs,static_cast<unsigned long long>(uint64_t(GpuParticleFieldCount)*sizeof(GpuParticleFieldState)));
    for(auto pool:{std::span<Timing>(timings),std::span<Timing>(stepTimings),std::span<Timing>(publicationTimings),std::span<Timing>(particleTimings)})for(auto& t:pool){cudaEventDestroy(t.begin);cudaEventDestroy(t.end);}
    for(void* p:{static_cast<void*>(airflowDevice),static_cast<void*>(dragDevice),static_cast<void*>(indices),static_cast<void*>(previous),static_cast<void*>(current),static_cast<void*>(velocity),static_cast<void*>(angular),static_cast<void*>(forces),static_cast<void*>(torques),static_cast<void*>(visualDevice),static_cast<void*>(lightDevice),static_cast<void*>(aeroDevice),static_cast<void*>(probeDevice),static_cast<void*>(actorDevice),static_cast<void*>(contacts),static_cast<void*>(contactCount),static_cast<void*>(probeOutput)})if(p)cudaFree(p);
    for(auto e:{uploadReady,poseReady,velocityReady,stateReady,forceReady,forceDone,torqueDone,contactReady})if(e)cuEventDestroy(e);
    for(auto e:{populationInputsReady,populationPackReady})if(e)Cuda(cudaEventDestroy(e),"destroy population publication event");
    for(auto e:populationCopiesDone)if(e)Cuda(cudaEventDestroy(e),"destroy population group reuse event");
    if(populationStream)Cuda(cudaStreamDestroy(populationStream),"destroy independent population stream");
    if(snowfallStream)Cuda(cudaStreamDestroy(snowfallStream),"destroy independent snowfall stream");
    if(stream)cudaStreamDestroy(stream);}
    for(auto& j:joints)if(j.joint)j.joint->release();
    for(auto& b:bodies){if(b.actor)b.actor->release();if(b.material)b.material->release();if(b.convex)b.convex->release();}
    if(scene)scene->release();scene=nullptr;
}
void PhysicsRuntime::Impl::Scene::RefreshIndices(){
    indexHost.clear();for(auto& b:bodies){b.index=PhysicsInvalidIndex;if(!b.actor||!b.actor->is<PxRigidDynamic>())continue;auto index=static_cast<PxRigidDynamic*>(b.actor)->getGPUIndex();if(index==0xffffffffu)Fail("PhysX: missing rigid dynamic GPU index after simulation");b.index=uint32_t(indexHost.size());indexHost.push_back(index);}
    dynamicCount=uint32_t(indexHost.size());Upload(stream,indices,indexHost);topologyDirty=false;descriptorsDirty=true;
}
void PhysicsRuntime::Impl::Scene::RefreshDescriptors(){
    airflowHost.clear();dragHost.clear();visualHost.clear();lightHost.clear();aeroHost.clear();probeHost.clear();actorHost.clear();populationBodyHost.clear();
    for(const auto& f:airflows){auto* b=FindBody(f.emitter);if(b)airflowHost.push_back({f,b->index});}
    for(const auto& b:bodies)if(b.actor&&!b.kinematic&&b.index!=PhysicsInvalidIndex)dragHost.push_back({b.index,b.geometry});
    Upload(stream,airflowDevice,airflowHost);Upload(stream,dragDevice,dragHost);
    for(auto& v:visuals){auto* b=FindBody(v.body);if(!b)Fail("PhysX: expired visual body");visualHost.push_back({v,b->index,b->initial});}
    // Hash exactly the fields reaching RayBuildMain. Hashing a padded C++
    // object would make unrelated descriptor uploads appear to move physics.
    uint64_t signature=1469598103934665603ull;
    auto word=[&](uint32_t v){for(uint32_t shift=0;shift<32;shift+=8){signature^=(v>>shift)&255u;signature*=1099511628211ull;}};
    auto wide=[&](uint64_t v){word(uint32_t(v));word(uint32_t(v>>32));};
    auto real=[&](float v){word(std::bit_cast<uint32_t>(v));};
    physicalRayCanMove=false;
    for(const auto& item:visualHost){
        const auto& v=item.visual;
        word(v.mesh);word(v.material);word(v.id);word(v.flags);wide(v.body);
        word(item.index);
        real(v.localPose.position.x);real(v.localPose.position.y);real(v.localPose.position.z);
        real(v.localPose.orientation.x);real(v.localPose.orientation.y);
        real(v.localPose.orientation.z);real(v.localPose.orientation.w);
        real(v.scale.x);real(v.scale.y);real(v.scale.z);real(v.boundingRadius);
        real(item.initial.p.x);real(item.initial.p.y);real(item.initial.p.z);
        real(item.initial.q.x);real(item.initial.q.y);real(item.initial.q.z);real(item.initial.q.w);
        physicalRayCanMove|=item.index!=PhysicsInvalidIndex;
    }
    physicalDescriptorSignature=signature;
    for(auto& l:lights){auto* b=FindBody(l.body);if(!b)Fail("PhysX: expired light body");lightHost.push_back({l,b->index,b->initial});}
    for(auto& a:aero){auto* wing=FindBody(a.wing);auto* carrier=FindBody(a.carrier);if(wing&&carrier)aeroHost.push_back({a,wing->index,carrier->index});}
    for(auto h:probes){auto* b=FindBody(h);if(b)probeHost.push_back({h,reinterpret_cast<uint64_t>(b->actor),b->index,0,b->initial});}
    for(auto& b:bodies)if(b.actor)actorHost.push_back({reinterpret_cast<uint64_t>(b.actor),b.handle});
    for(auto& b:bodies)if(b.actor){const auto& g=b.geometry;PopulationBodyGpu d{};d.initial=b.initial;d.local=P(g.shapePose);d.dimensions=V(g.dimensions);d.index=b.index;d.shape=g.shape;d.enabled=b.collisionEnabled&&(g.collisionMask&population.layer)&&(g.layer&population.collisionMask);d.hand=b.handle==populationInput.hands[0]||b.handle==populationInput.hands[1];d.kinematic=b.kinematic;d.restitution=g.restitution;d.handle=b.handle;
        // Flies use a conservative sphere around a convex PhysX collider.
        // The rigid bodies keep their cooked convex geometry and exact contacts.
        if(g.shape==LAB_PHYSICS_CONVEX){d.shape=LAB_PHYSICS_SPHERE;d.dimensions=PxVec3(V(g.dimensions).magnitude());}
        d.radius=d.shape==LAB_PHYSICS_BOX?d.dimensions.magnitude():d.dimensions.x+(d.shape==LAB_PHYSICS_CAPSULE?d.dimensions.y:0);populationBodyHost.push_back(d);}
    Upload(stream,populationBodies,populationBodyHost);
    Upload(stream,visualDevice,visualHost);Upload(stream,lightDevice,lightHost);Upload(stream,aeroDevice,aeroHost);Upload(stream,probeDevice,probeHost);Upload(stream,actorDevice,actorHost);descriptorsDirty=false;
}
void PhysicsRuntime::Impl::Scene::ReadState(bool first){
    if(!dynamicCount)return;Driver(cuEventRecord(uploadReady,reinterpret_cast<CUstream>(stream)),"stage direct GPU indices");auto& api=scene->getDirectGPUAPI();
    Gpu(api.getRigidDynamicData(current,indices,PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE,dynamicCount,uploadReady,poseReady),"PhysX direct GPU pose publication failed");
    Gpu(api.getRigidDynamicData(velocity,indices,PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY,dynamicCount,poseReady,velocityReady),"PhysX direct GPU velocity read failed");
    Gpu(api.getRigidDynamicData(angular,indices,PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY,dynamicCount,velocityReady,stateReady),"PhysX direct GPU angular read failed");
    Driver(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream),stateReady,0),"wait grouped physical state");
    if(first)Cuda(cudaMemcpyAsync(previous,current,dynamicCount*sizeof(PxTransform),cudaMemcpyDeviceToDevice,stream),"initialize coherent previous poses");
}
PhysicsRuntime::Impl::Scene::Timing* PhysicsRuntime::Impl::Scene::BeginTiming(std::span<Timing> pool,uint32_t& cursor,uint64_t sampleTick,uint64_t value,cudaStream_t executionStream){
    // No CUDA profiler events in live play or the quiet control window.
    // The simulation and the publication fences are not altered.
    if(!SessionLogCompact())return nullptr;
    for(uint32_t i=0;i<pool.size();i++){
        auto index=(cursor+i)%uint32_t(pool.size());auto& timing=pool[index];
        if(timing.pending)continue;
        timing.frame=frame;timing.tick=sampleTick;timing.value=value;timing.pending=true;
        Cuda(cudaEventRecord(timing.begin,executionStream?executionStream:stream),"begin physical GPU span");
        cursor=(index+1)%uint32_t(pool.size());return &timing;
    }
    return nullptr; // Profiling never adds a blocking wait to the simulation.
}
void PhysicsRuntime::Impl::Scene::CollectTimings(){
    auto collect=[&](std::span<Timing> pool,const char* stage,uint32_t kind){
        for(auto& t:pool)if(t.pending){
            auto ready=cudaEventQuery(t.end);if(ready==cudaErrorNotReady)continue;
            Cuda(ready,"query physical GPU timing");float span{};
            Cuda(cudaEventElapsedTime(&span,t.begin,t.end),"measure physical GPU span");
            // Non-blocking event harvest: spans include CUDA stream dependencies
            // and GPU scheduling gaps, not exclusively PhysX kernel execution.
            if(SessionLogCompact()){
                if(kind==1){
                    ++diagnostic.gpuStepSamples;
                    diagnostic.gpuStepSumMs+=span;
                    diagnostic.gpuStepPeakMs=std::max(diagnostic.gpuStepPeakMs,double(span));
                }else if(kind==2){
                    ++diagnostic.gpuFrameSamples;
                    diagnostic.gpuFrameSumMs+=span;
                    diagnostic.gpuFramePeakMs=std::max(diagnostic.gpuFramePeakMs,double(span));
                }else if(kind==3){
                    ++particleSamples;
                    particleSumMs+=span;
                    particlePeakMs=std::max(particlePeakMs,double(span));
                }
            }
            if(kind!=3)LogEvent(t.frame,SessionLogCompact()&&kind==2?"physics.profile.frame_gpu":stage,"scene=%llu tick=%llu fence_value=%llu gpu_span_ms=%.4f",static_cast<unsigned long long>(handle),static_cast<unsigned long long>(t.tick),static_cast<unsigned long long>(t.value),span);
            t.pending=false;
        }
    };
    collect(timings,"physics.gpu",2);
    collect(stepTimings,"physics.step.gpu",1);
    collect(publicationTimings,"physics.publication.wait",0);
    collect(particleTimings,"physics.particles.gpu",3);
}
void PhysicsRuntime::Impl::Scene::Step(float targetFraction,uint32_t stepOrdinal){
    const double stepBegin=QpcNowSeconds();double commandWaitMs=0,topologyWaitMs=0,uploadSubmitMs=0,populationSubmitMs=0;
    bool forceDeviceCleared=false,forceHostUploaded=false;
    auto* timing=BeginTiming(stepTimings,stepTimingCursor,tick+1);
    if(!timing)++frameUntimedSteps;
    const bool rebuild=topologyDirty;
    if(started&&!rebuild&&dynamicCount){
        // No explicit physical force is the common case: kinematic targets,
        // airflow and aero all have their own inputs or GPU kernels. Do not
        // clear pinned CPU arrays and upload them across PCIe for zero data.
        // Keep the old host accumulation path for force/torque commands.
        for(const auto& c:commands){
            if(c.kind==LAB_PHYSICS_KINEMATIC_TARGET)continue;
            auto* b=FindBody(c.body);if(!b||b->index==PhysicsInvalidIndex)continue;
            if(!forceHostUploaded){
                std::fill_n(forceHost,dynamicCount,PxVec3(0));
                std::fill_n(torqueHost,dynamicCount,PxVec3(0));
                forceHostUploaded=true;
            }
            const float multiplier=(c.kind==LAB_PHYSICS_IMPULSE||c.kind==LAB_PHYSICS_ANGULAR_IMPULSE)?1.f/desc.fixedStepSeconds:1.f;
            if(c.kind==LAB_PHYSICS_FORCE||c.kind==LAB_PHYSICS_IMPULSE)forceHost[b->index]+=V(c.value)*multiplier;
            else torqueHost[b->index]+=V(c.value)*multiplier;
        }
        const double uploadBegin=QpcNowSeconds();
        const size_t forceBytes=size_t(dynamicCount)*sizeof(PxVec3);
        if(forceHostUploaded){
            Cuda(cudaMemcpyAsync(forces,forceHost,forceBytes,cudaMemcpyHostToDevice,stream),"upload grouped forces");
            Cuda(cudaMemcpyAsync(torques,torqueHost,forceBytes,cudaMemcpyHostToDevice,stream),"upload grouped torques");
        }else{
            // Same GPU stream orders the zeroing before insect/airflow/aero
            // kernels and before PhysX Direct GPU API consumes the arrays.
            Cuda(cudaMemsetAsync(forces,0,forceBytes,stream),"reset GPU forces");
            Cuda(cudaMemsetAsync(torques,0,forceBytes,stream),"reset GPU torques");
            forceDeviceCleared=true;
        }
        uploadSubmitMs=(QpcNowSeconds()-uploadBegin)*1000;
        const double populationBegin=SessionLogCompact()?QpcNowSeconds():0.;
        StepPopulation();
        if(populationBegin>0)populationSubmitMs+=(QpcNowSeconds()-populationBegin)*1000.;
        PhysicsApplyAirflow(stream,dragDevice,uint32_t(dragHost.size()),airflowDevice,uint32_t(airflowHost.size()),current,velocity,angular,forces,torques,float(simulationTime));Cuda(cudaGetLastError(),"rotor airflow forces");
        PhysicsApplyAero(stream,aeroDevice,uint32_t(aeroHost.size()),current,velocity,angular,forces,torques);Cuda(cudaGetLastError(),"wing aerodynamic commands");
        Driver(cuEventRecord(forceReady,reinterpret_cast<CUstream>(stream)),"publish grouped commands");auto& api=scene->getDirectGPUAPI();
        Gpu(api.setRigidDynamicData(forces,indices,PxRigidDynamicGPUAPIWriteType::eFORCE,dynamicCount,forceReady,forceDone),"PhysX direct GPU forces rejected");
        Gpu(api.setRigidDynamicData(torques,indices,PxRigidDynamicGPUAPIWriteType::eTORQUE,dynamicCount,forceDone,torqueDone),"PhysX direct GPU torques rejected");
        // PhysX simulate has no external start-event argument. One batch event
        // gates the submission, never one wait per actor or a graphics WaitIdle.
        auto ready=cuEventQuery(torqueDone);
        if(ready==CUDA_ERROR_NOT_READY){
            double waitBegin=QpcNowSeconds();Driver(cuEventSynchronize(torqueDone),"complete grouped simulation commands");
            commandWaitMs=(QpcNowSeconds()-waitBegin)*1000;
        }else Driver(ready,"query grouped simulation commands");
        Cuda(cudaMemcpyAsync(previous,current,dynamicCount*sizeof(PxTransform),cudaMemcpyDeviceToDevice,stream),"retain previous fixed-step poses");
        // forceReady follows the previous probe/visual kernels on this stream,
        // so torqueDone already retires all consumers of PhysX internal contact
        // pointers. The external previous-pose copy can overlap simulate(); the
        // next GPU state read waits on uploadReady recorded after that copy.
    }else {
        double waitBegin=QpcNowSeconds();Cuda(cudaStreamSynchronize(stream),"prepare scene topology");
        topologyWaitMs=(QpcNowSeconds()-waitBegin)*1000;
    }
    if(started&&!rebuild&&!dynamicCount){
        const double populationBegin=SessionLogCompact()?QpcNowSeconds():0.;
        StepPopulation();
        if(populationBegin>0)populationSubmitMs+=(QpcNowSeconds()-populationBegin)*1000.;
    }
    for(auto& c:commands)if(c.kind==LAB_PHYSICS_KINEMATIC_TARGET){auto* b=FindBody(c.body);if(b&&b->kinematic){auto end=P(c.target);PxQuat q=end.q;if(b->lastTarget.q.dot(q)<0)q=-q;PxTransform target(b->lastTarget.p+(end.p-b->lastTarget.p)*targetFraction,(b->lastTarget.q*(1-targetFraction)+q*targetFraction).getNormalized());static_cast<PxRigidDynamic*>(b->actor)->setKinematicTarget(target);}}
    for(auto& j:joints)if(j.joint&&j.motor.structSize){
        auto target=P(j.motor.target);auto omega=V(j.motor.angularVelocity);
        if(j.motor.frequencyHz>0){
            // Frequency changes alter the next cycle's rate, never its current
            // phase. These are D6 motor targets; PhysX alone moves the bodies.
            constexpr double tau=6.283185307179586;
            double phase=j.motorPhase+double(j.motor.phaseRadians);
            float angle=j.motor.amplitudeRadians*float(std::sin(phase));target.q=PxQuat(angle,PxVec3(1,0,0));
            omega.x=j.motor.amplitudeRadians*j.motor.frequencyHz*float(tau)*float(std::cos(phase));
            j.motorPhase=std::fmod(j.motorPhase+double(j.motor.frequencyHz)*tau*desc.fixedStepSeconds,tau);
        }
        const PxVec3 linear=V(j.motor.linearVelocity);
        const bool unchanged=j.motor.frequencyHz==0&&j.lastDriveValid
            &&target.p.x==j.lastDrivePosition.p.x
            &&target.p.y==j.lastDrivePosition.p.y
            &&target.p.z==j.lastDrivePosition.p.z
            &&target.q.x==j.lastDrivePosition.q.x
            &&target.q.y==j.lastDrivePosition.q.y
            &&target.q.z==j.lastDrivePosition.q.z
            &&target.q.w==j.lastDrivePosition.q.w
            &&linear.x==j.lastDriveLinear.x
            &&linear.y==j.lastDriveLinear.y
            &&linear.z==j.lastDriveLinear.z
            &&omega.x==j.lastDriveAngular.x
            &&omega.y==j.lastDriveAngular.y
            &&omega.z==j.lastDriveAngular.z;
        if(unchanged){
            if(SessionLogCompact())++diagnostic.motorUnchangedSkips;
            continue;
        }
        j.joint->setDrivePosition(target,false);
        j.joint->setDriveVelocity(linear,omega,false);
        j.lastDrivePosition=target;
        j.lastDriveLinear=linear;
        j.lastDriveAngular=omega;
        j.lastDriveValid=true;
        if(SessionLogCompact())++diagnostic.motorWrites;
    }
    double simulateBegin=QpcNowSeconds();scene->simulate(desc.fixedStepSeconds);
    double simulateMs=(QpcNowSeconds()-simulateBegin)*1000;
    const double checkBegin=SessionLogCompact()?QpcNowSeconds():0.;
    bool resultsReady=scene->checkResults(false);PxU32 error=0;
    const double checkMs=checkBegin>0?(QpcNowSeconds()-checkBegin)*1000.:0.;
    double fetchBegin=QpcNowSeconds();
    // Serial fixed steps require completed PhysX results before another simulate
    // or direct-GPU state query. Keep that required rendezvous explicit; do not
    // request the blocking path when checkResults already establishes readiness.
    Gpu(scene->fetchResults(!resultsReady,&error)&&error==0,"PhysX GPU simulation failed; no CPU fallback allowed");
    double fetchMs=(QpcNowSeconds()-fetchBegin)*1000;runtime.Check();
    const double postFetchBegin=SessionLogCompact()?QpcNowSeconds():0.;
    if(rebuild)RefreshIndices();if(descriptorsDirty)RefreshDescriptors();ReadState(!started||rebuild);started=true;++tick;simulationTime+=desc.fixedStepSeconds;
    const double postFetchMs=postFetchBegin>0?(QpcNowSeconds()-postFetchBegin)*1000.:0.;
    if(timing)Cuda(cudaEventRecord(timing->end,stream),"end physical fixed-step GPU span");
    frameCommandWaitMs+=commandWaitMs;frameFetchBlockingMs+=resultsReady?0:fetchMs;frameTopologyWaitMs+=topologyWaitMs;
    const double stepCpuMs=(QpcNowSeconds()-stepBegin)*1000.;
    if(SessionLogCompact()){
        auto& p=diagnostic;
        ++p.steps;p.stepCpuMs+=stepCpuMs;p.maxStepCpuMs=std::max(p.maxStepCpuMs,stepCpuMs);
        p.commandWaitMs+=commandWaitMs;p.topologyWaitMs+=topologyWaitMs;
        p.fetchBlocked+=!resultsReady;
        const uint32_t ordinal=std::min(stepOrdinal,1u);
        ++p.fetchByOrdinal[ordinal];p.fetchSumByOrdinalMs[ordinal]+=fetchMs;
        p.fetchPeakByOrdinalMs[ordinal]=std::max(p.fetchPeakByOrdinalMs[ordinal],fetchMs);
        ++p.fetchBuckets[fetchMs<0.5?0:fetchMs<1.?1:fetchMs<2.?2:fetchMs<4.?3:4];
        p.checkResultsMs+=checkMs;
        if(resultsReady)p.fetchReadyMs+=fetchMs;else p.fetchBlockedMs+=fetchMs;
        p.simulateSubmitMs+=simulateMs;p.populationSubmitMs+=populationSubmitMs;p.postFetchCpuMs+=postFetchMs;
        p.deviceZeroSteps+=forceDeviceCleared;
        p.hostUploadSteps+=forceHostUploaded;
        p.untimedGpuSteps+=timing==nullptr;
        p.forceInitSubmitMs+=uploadSubmitMs;
    }
    LogEvent(frame,"physics.step","scene=%llu tick=%llu cpu_ms=%.4f command_upload_submit_ms=%.4f command_wait_ms=%.4f topology_wait_ms=%.4f simulate_submit_ms=%.4f fetch_ms=%.4f fetch_blocking=%u gpu_timed=%u",static_cast<unsigned long long>(handle),static_cast<unsigned long long>(tick),stepCpuMs,uploadSubmitMs,commandWaitMs,topologyWaitMs,simulateMs,fetchMs,unsigned(!resultsReady),unsigned(timing!=nullptr));
    if(started&&!rebuild)std::erase_if(commands,[](const auto& c){return c.kind==LAB_PHYSICS_IMPULSE||c.kind==LAB_PHYSICS_ANGULAR_IMPULSE;});
}
void PhysicsRuntime::Impl::Scene::Advance(double now,uint64_t inputFrame){
    double begin=QpcNowSeconds();PxScopedCudaLock lock(*runtime.cuda);frame=inputFrame;
    double contextLockMs=(QpcNowSeconds()-begin)*1000;
    frameCommandWaitMs=frameFetchBlockingMs=frameTopologyWaitMs=0;frameUntimedSteps=0;
    if(paused){lastTime=now;accumulator=0;alpha=0;commands.clear();if(descriptorsDirty&&started)RefreshDescriptors();return;}
    CollectTimings();auto* timing=BeginTiming(timings,timingCursor);
    if(insectCapacity&&!populationClockStarted){populationClockOrigin=now;populationClockStarted=true;}
    double elapsed=lastTime>0?std::max(0.,now-lastTime):double(desc.fixedStepSeconds);lastTime=now;
    const double budget=double(desc.fixedStepSeconds)*desc.maxSubsteps;
    const double droppedBefore=droppedSeconds;
    // Never carry whole overdue steps into later frames. A CPU/GPU stall
    // must not create extra PhysX fetchResults waits on the next frame.
    // Retain only the fractional remainder needed to align 72/90 Hz XR
    // frames with the fixed 120 Hz physics clock.
    const double owed=accumulator+elapsed;
    // When two steps are sufficient, keep their fractional remainder untouched.
    // Only whole extra steps are discarded; otherwise even healthy 72 Hz XR
    // would lose simulation time at every alternate frame.
    double discarded=0.;
    if(owed>=budget+double(desc.fixedStepSeconds))
        discarded=std::floor((owed-budget)/double(desc.fixedStepSeconds))*double(desc.fixedStepSeconds);
    droppedSeconds+=discarded;
    accumulator=owed-discarded;
    populationSeconds+=std::max(0.,elapsed-discarded);uint32_t steps=0;bool forceCommandsConsumed=false;
    uint32_t planned=std::min(desc.maxSubsteps,uint32_t((accumulator+1e-9)/desc.fixedStepSeconds));
    while(accumulator+1e-9>=desc.fixedStepSeconds&&steps<desc.maxSubsteps){
        // Topology rebuild steps cannot upload force commands: keep them until
        // a later step has valid Direct GPU indices.
        forceCommandsConsumed|=started&&!topologyDirty&&dynamicCount>0;
        Step(float(steps+1)/float(std::max(1u,planned)),steps);accumulator-=desc.fixedStepSeconds;++steps;
    }
    alpha=float(std::clamp(accumulator/double(desc.fixedStepSeconds),0.,1.));
    // Walk-simulator snowfall: one elapsed-time CUDA update per displayed XR
    // frame. Do not run snow inside the 120 Hz rigid-body substep loop.
    // Keep actual elapsed weather time, but defer work until an XR image is
    // acquired. The CUDA integration runs directly into that image's shared
    // buffer, saving the separate 64 MB device-to-device publication copy.
    if(snowfallReady){
        snowfallPendingSeconds=std::min(.05,snowfallPendingSeconds+std::clamp(elapsed,0.,.05));
        snowfallTimeSeconds=now;
    }
    if(descriptorsDirty&&started){Cuda(cudaStreamSynchronize(stream),"retire descriptor uploads");RefreshDescriptors();}
    if(started)QueueProbe();
    if(timing)Cuda(cudaEventRecord(timing->end,stream),"end physical GPU span");
    // Continuous commands expire only after being consumed. An impulse survives
    // a display frame that had no physics tick.
    if(steps){for(const auto& c:commands)if(c.kind==LAB_PHYSICS_KINEMATIC_TARGET){if(auto* b=FindBody(c.body))b->lastTarget=P(c.target);}}
    if(steps)std::erase_if(commands,[forceCommandsConsumed](const auto& c){return c.kind==LAB_PHYSICS_KINEMATIC_TARGET||(forceCommandsConsumed&&c.kind!=LAB_PHYSICS_IMPULSE&&c.kind!=LAB_PHYSICS_ANGULAR_IMPULSE);});
    const double frameCpuMs=(QpcNowSeconds()-begin)*1000.;
    if(SessionLogCompact()){
        ++diagnostic.frames;diagnostic.droppedFrames+=droppedSeconds>droppedBefore;
        diagnostic.frameCpuMs+=frameCpuMs;diagnostic.contextLockMs+=contextLockMs;
    }
    LogEvent(frame,SessionLogCompact()?"physics.profile.frame":"physics.frame","scene=%llu tick=%llu steps=%u fixed_dt=%.8f cpu_ms=%.4f context_lock_ms=%.4f command_wait_ms=%.4f fetch_blocking_call_ms=%.4f topology_wait_ms=%.4f untimed_gpu_steps=%u frame_gpu_timed=%u publication_backpressure_total=%llu dropped_seconds=%.6f probe_count=%u readback_bytes=%u",static_cast<unsigned long long>(handle),static_cast<unsigned long long>(tick),steps,desc.fixedStepSeconds,frameCpuMs,contextLockMs,frameCommandWaitMs,frameFetchBlockingMs,frameTopologyWaitMs,frameUntimedSteps,unsigned(timing!=nullptr),static_cast<unsigned long long>(publicationBackpressureCount),droppedSeconds,uint32_t(probeHost.size()),uint32_t(probeHost.size()*sizeof(LabPhysicsStateV1)+sizeof(uint32_t)));
}
uint32_t PhysicsRuntime::Impl::Scene::SetVisuals(const LabPhysicsVisualV1* v,uint32_t n,const LabPhysicsLightV1* l,uint32_t nl){
    if(n>MaxVisuals||nl>MaxLights||(n&&!v)||(nl&&!l))return 0;
    for(uint32_t i=0;i<n;i++)if(v[i].structSize<sizeof(*v)||!FindBody(v[i].body)||v[i].mesh>=ConsoleSpec::MaxMeshes||v[i].material>=ConsoleSpec::MaxMaterials||!Valid(v[i].localPose)||!Valid(v[i].scale)||!Finite(v[i].boundingRadius)||v[i].boundingRadius<=0)return 0;
    for(uint32_t i=0;i<nl;i++)if(l[i].structSize<sizeof(*l)||!FindBody(l[i].body)||l[i].sceneLightIndex>=ConsoleSpec::MaxLights||!Valid(l[i].localPosition)||!Valid(l[i].localDirection)||!Valid(l[i].color)||!Finite(l[i].radius)||l[i].radius<=0||!Finite(l[i].power)||l[i].power<0||!Finite(l[i].outerCos)||!Finite(l[i].innerCos)||l[i].outerCos<-1||l[i].outerCos>1||l[i].innerCos<l[i].outerCos||l[i].innerCos>1)return 0;
    PxScopedCudaLock lock(*runtime.cuda);
    uint32_t publicationCapacity=n+(insectCapacity?insectCapacity+4*LAB_PHYSICS_POPULATION_DETAIL_MAX_V1+LAB_POPULATION_SHADOW_MAX_V1:0);
    // Snowfall can be the only published GPU domain. Reserve a control
    // buffer even when no physical visuals and no insect population exist.
    if(snowfallReady)publicationCapacity=std::max(publicationCapacity,1u);
    bool sameLayout=n==visuals.size()&&nl==lights.size()&&!slots[0].refs.empty()&&slots[0].instances.bytes==uint64_t(publicationCapacity)*sizeof(GpuInstance);
    for(uint32_t i=0;i<nl&&sameLayout;i++)sameLayout=lights[i].sceneLightIndex==l[i].sceneLightIndex;
    for(uint32_t i=0;i<n&&sameLayout;i++)sameLayout=std::find(streamMaterials.begin(),streamMaterials.end(),v[i].material)!=streamMaterials.end();
    if(sameLayout){
        // Descriptor/material/light edits preserve the shared ring and fences.
        // Retire prior descriptor consumers once, then batch the new upload.
        Cuda(cudaStreamSynchronize(stream),"retire visual descriptor edits");
        if(n)visuals.assign(v,v+n);if(nl)lights.assign(l,l+nl);descriptorsDirty=true;return 1;
    }
    DestroyPublications();visuals.clear();lights.clear();if(n)visuals.assign(v,v+n);if(nl)lights.assign(l,l+nl);streamMaterials.clear();
    for(const auto& x:visuals)if(std::find(streamMaterials.begin(),streamMaterials.end(),x.material)==streamMaterials.end())streamMaterials.push_back(x.material);
    if(insectCapacity)for(auto mat:{population.bodyMaterial,population.headMaterial,population.wingMaterial})if(std::find(streamMaterials.begin(),streamMaterials.end(),mat)==streamMaterials.end())streamMaterials.push_back(mat);
    provider.streamMaterials=streamMaterials.data();provider.streamMaterialCount=uint32_t(streamMaterials.size());
    if(publicationCapacity)for(uint32_t slot=0;slot<slots.size();++slot){
        auto& s=slots[slot];
        s.instances.Create(owner.interop,uint64_t(publicationCapacity)*sizeof(GpuInstance));
        s.control.Create(owner.interop,48);
        if(nl)s.physicsVisual.lighting.Create(owner.interop,uint64_t(nl)*sizeof(PhysicsGpuLight));
        // Independent backing allocations are populated after the existing pack
        // kernels. Rendering still consumes the exact original combined range.
        if(n){
            s.physicsVisual.instances.Create(owner.interop,uint64_t(n)*sizeof(GpuInstance));
            s.physicsVisual.storageGeneration=runtime.LeaseGeneration();
        }
        if(n||nl)s.physicsVisual.control.Create(owner.interop,48);
        if(insectCapacity){
            s.population.instances.Create(owner.interop,uint64_t(insectCapacity+4*LAB_PHYSICS_POPULATION_DETAIL_MAX_V1+LAB_POPULATION_SHADOW_MAX_V1)*sizeof(GpuInstance));
            s.population.control.Create(owner.interop,48);
        }
        s.producer.Create(owner.interop);s.consumer.Create(owner.interop);
        s.submission=std::make_shared<GpuProducerSubmission>();
        if(n||nl){s.physicsVisual.producer.Create(owner.interop);s.physicsVisual.consumer.Create(owner.interop);s.physicsVisual.submission=std::make_shared<GpuProducerSubmission>();}
        if(insectCapacity){s.population.producer.Create(owner.interop);s.population.consumer.Create(owner.interop);s.population.submission=std::make_shared<GpuProducerSubmission>();}
        if(snowfallReady){s.snow.producer.Create(owner.interop);s.snow.consumer.Create(owner.interop);s.snow.submission=std::make_shared<GpuProducerSubmission>();}
        auto ref=[&](uint32_t binding,const SharedBuffer& buffer,uint64_t offset,uint64_t bytes,PublicationSlot::ResourceGroup* group=nullptr){
            LabGpuResourceRefV1 r{};r.binding=binding;r.kind=LABOVR_GPU_RESOURCE_BUFFER_V1;r.resourceHandle=buffer.handle;r.bytes=bytes;r.offset=offset;r.generation=handle;r.initialLayout=uint32_t(D3D12_BARRIER_LAYOUT_COMMON);s.refs.push_back(r);
            if(group)group->bindings.push_back(binding);
        };
        ref(0,s.instances,0,s.instances.bytes);ref(1,s.control,0,16);ref(2,s.control,16,32);
        if(insectCapacity){
            ref(LAB_POPULATION_OPTICAL_BINDING_V1,s.population.instances,0,sizeof(GpuInstance),&s.population);
            ref(LAB_POPULATION_SHADOW_BINDING_V1,s.population.instances,0,LAB_POPULATION_SHADOW_MAX_V1*sizeof(GpuInstance),&s.population);
            ref(LAB_POPULATION_PROXY_BINDING_V1,s.population.instances,0,sizeof(GpuInstance),&s.population);
        }
        if(snowfallReady)ref(LAB_SNOW_FIELD_BINDING_V1,snowfallStates[slot],0,snowfallStates[slot].bytes);
        for(uint32_t i=0;i<nl;i++)ref(LABOVR_GPU_MESH_BINDING_LIGHT_BASE_V1+lights[i].sceneLightIndex,s.physicsVisual.lighting,uint64_t(i)*sizeof(PhysicsGpuLight),sizeof(PhysicsGpuLight),&s.physicsVisual);
        if(n)ref(GpuMeshPublicationAbi::PhysicsInstances,s.physicsVisual.instances,0,s.physicsVisual.instances.bytes,&s.physicsVisual);
        if(n||nl)ref(GpuMeshPublicationAbi::PhysicsControl,s.physicsVisual.control,0,s.physicsVisual.control.bytes,&s.physicsVisual);
        if(insectCapacity){
            ref(GpuMeshPublicationAbi::PopulationInstances,s.population.instances,0,s.population.instances.bytes,&s.population);
            ref(GpuMeshPublicationAbi::PopulationControl,s.population.control,0,s.population.control.bytes,&s.population);
        }
        if(snowfallReady)ref(GpuMeshPublicationAbi::SnowState,snowfallStates[slot],0,snowfallStates[slot].bytes,&s.snow);
    }
    descriptorsDirty=true;return 1;
}
uint32_t PhysicsRuntime::Impl::Scene::ReservePublication(LabGpuPublicationV1* out,bool deferred){
    if(!out||out->structSize<sizeof(*out)||out->gpuAbiVersion!=LABOVR_GPU_ABI_V1||(visuals.empty()&&!insectCapacity&&!snowfallReady)||!started||topologyDirty||pendingPublication)return 0;
    // Descriptor rebuilds can change the publication layout during Advance;
    // use the existing synchronous path for those uncommon frames.
    if(deferred&&descriptorsDirty)return 0;
    PxScopedCudaLock lock(*runtime.cuda);if(descriptorsDirty)RefreshDescriptors();
    PublicationSlot* chosen{};PublicationSlot* available{};uint32_t chosenIndex{},availableIndex{};
    for(uint32_t i=0;i<slots.size();i++){
        uint32_t index=(publicationCursor+i)%uint32_t(slots.size());auto& slot=slots[index];
        if(slot.leased)continue;
        if(!available){available=&slot;availableIndex=index;}
        uint64_t value=slot.consumer.Completed(owner.interop);
        if(value>=slot.value){chosen=&slot;chosenIndex=index;break;}
    }
    if(!chosen&&available){chosen=available;chosenIndex=availableIndex;}
    // CPU lease ownership and GPU completion are distinct. Release means the
    // previous consumer signal has been enqueued, so an unleased slot can be
    // reused now with a GPU wait. A pending DX12 fence is never an Acquire failure.
    if(!chosen){LogEvent(frame,"physics.publication","scene=%llu leases_exhausted=1",static_cast<unsigned long long>(handle));return 0;}
    publicationCursor=(chosenIndex+1)%uint32_t(slots.size());auto& s=*chosen;
    if(insectCapacity){
        s.refs[0].bytes=(uint64_t(visualHost.size())+insectCount+4*LAB_PHYSICS_POPULATION_DETAIL_MAX_V1+LAB_POPULATION_SHADOW_MAX_V1)*sizeof(GpuInstance);
        for(auto& ref:s.refs){
            if(ref.binding==LAB_POPULATION_SHADOW_BINDING_V1)ref.offset=(uint64_t(insectCount)+4*LAB_PHYSICS_POPULATION_DETAIL_MAX_V1)*sizeof(GpuInstance);
            if(ref.binding==LAB_POPULATION_PROXY_BINDING_V1||ref.binding==LAB_POPULATION_OPTICAL_BINDING_V1){ref.offset=0;ref.bytes=uint64_t(std::max(1u,insectCount))*sizeof(GpuInstance);ref.flags=insectCount?0:1;}
        }
    }
    // Reserve each group from its own ring. Reservation only leases CPU
    // ownership; GPU reuse waits for that group's prior consumer value.
    const std::array<bool,3> activeGroups={ !visuals.empty()||!lights.empty(),insectCapacity!=0,snowfallReady };
    std::array<uint32_t,3> indices{};
    for(uint32_t kind=0;kind<3;kind++)if(activeGroups[kind]){
        bool found=false;
        for(uint32_t i=0;i<slots.size();i++){
            uint32_t index=(groupCursors[kind]+i)%uint32_t(slots.size());
            if(Group(kind,index).leased)continue;
            indices[kind]=index;found=true;break;
        }
        // GPU waits below, not CPU fence queries, arbitrate reuse.
        if(!found)return 0;
    }
    if(s.value==UINT64_MAX)Fail("PhysX publication fence exhausted");
    // Deferred reservations happen before the worker's PhysX step. A dynamic
    // visual may change between this reservation and GPU packing; conservatively
    // revise on each physical publication, never merely on snow/fly updates.
    const bool rayMayChange=!physicalRayPublished||
        publishedPhysicalDescriptorSignature!=physicalDescriptorSignature||
        physicalRayVisible!=visible||(physicalRayCanMove&&visible);
    if(rayMayChange){
        if(physicalRayRevision==UINT64_MAX)Fail("PhysX ray publication revision exhausted");
        ++physicalRayRevision;
        publishedPhysicalDescriptorSignature=physicalDescriptorSignature;
        physicalRayVisible=visible;
        physicalRayPublished=true;
    }
    s.physicalCount=uint32_t(visualHost.size());
    s.rayRevision=physicalRayRevision;
    for(uint32_t kind=0;kind<3;kind++)if(activeGroups[kind]&&Group(kind,indices[kind]).value==UINT64_MAX)
        Fail("PhysX GPU group fence exhausted");
    pendingConsumerValue=s.value;pendingPublication=chosen;pendingPublicationIndex=chosenIndex;
    ++s.value;s.leased=true;s.leaseGeneration=runtime.LeaseGeneration();
    s.groupActive=activeGroups;
    for(uint32_t kind=0;kind<3;kind++)if(activeGroups[kind]){
        auto& g=Group(kind,indices[kind]);s.groupSlots[kind]=indices[kind];
        s.groupPriorConsumerValues[kind]=g.value;++g.value;g.leased=true;g.leaseGeneration=s.leaseGeneration;
        groupCursors[kind]=(indices[kind]+1)%uint32_t(slots.size());
    }
    // The published binding table selects the independently reserved backing
    // resource while the legacy combined rendering indices remain unchanged.
    for(auto& ref:s.refs){
        if(ref.binding==GpuMeshPublicationAbi::PhysicsInstances)
            ref.resourceHandle=Group(0,indices[0]).instances.handle;
        else if(ref.binding==GpuMeshPublicationAbi::PhysicsControl)
            ref.resourceHandle=Group(0,indices[0]).control.handle;
        else if(ref.binding==GpuMeshPublicationAbi::PopulationInstances){
            ref.resourceHandle=Group(1,indices[1]).instances.handle;
            ref.bytes=(uint64_t(insectCount)+4*LAB_PHYSICS_POPULATION_DETAIL_MAX_V1+LAB_POPULATION_SHADOW_MAX_V1)*sizeof(GpuInstance);
        }else if(ref.binding==GpuMeshPublicationAbi::PopulationControl)
            ref.resourceHandle=Group(1,indices[1]).control.handle;
        else if(ref.binding==LAB_POPULATION_OPTICAL_BINDING_V1||
                ref.binding==LAB_POPULATION_SHADOW_BINDING_V1||
                ref.binding==LAB_POPULATION_PROXY_BINDING_V1)
            ref.resourceHandle=Group(1,indices[1]).instances.handle;
        else if(ref.binding==LAB_SNOW_FIELD_BINDING_V1||ref.binding==GpuMeshPublicationAbi::SnowState)
            ref.resourceHandle=snowfallStates[indices[2]].handle;
        else if(ref.binding>=GpuMeshPublicationAbi::LightBase&&ref.binding<GpuMeshPublicationAbi::LightBase+MaxLights)
            ref.resourceHandle=Group(0,indices[0]).lighting.handle;
    }
    *out={LABOVR_GPU_ABI_V1,sizeof(*out),LABOVR_GPU_DOMAIN_MESH_SOLID_V1,0,s.refs.data(),uint32_t(visible?s.refs.size():3),0,++publicationSequence,s.producer.handle,s.value,s.consumer.handle,s.value,&s,s.leaseGeneration};
    return 1;
}
void PhysicsRuntime::Impl::Scene::CompletePublication(){
    if(!pendingPublication)return;
    PxScopedCudaLock lock(*runtime.cuda);
    auto& s=*pendingPublication;const uint32_t chosenIndex=pendingPublicationIndex;
    // Snow has no rigid-body input: enqueue on its own non-blocking stream
    // before packing physics/population, without an inter-stream join.
    if(snowfallReady)QueueSnowfall();
    if(insectCapacity)QueuePopulationPublication();
    // Physics visual storage is independent of the legacy combined ring.
    // Produce and signal it before waiting for consumers of the combined
    // instances, which can include the previous frame's snowfall work.
    if(s.groupActive[0]){
        auto& group=Group(0,s.groupSlots[0]);
        group.consumer.Wait(stream,s.groupPriorConsumerValues[0]);
        PhysicsPackVisuals(stream,visualDevice,uint32_t(visualHost.size()),lightDevice,uint32_t(lightHost.size()),previous,current,alpha,visible,
            static_cast<GpuInstance*>(group.instances.pointer),static_cast<uint32_t*>(group.control.pointer),
            static_cast<PhysicsGpuLight*>(group.lighting.pointer));
        Cuda(cudaGetLastError(),"pack independent GPU physical visuals");
        group.producer.Signal(stream,group.value);
        group.submission->SignalEnqueued(group.value);
    }
    const uint64_t completed=s.consumer.Completed(owner.interop);
    Timing* waitTiming=nullptr;
    if(completed<pendingConsumerValue){
        ++publicationBackpressureCount;waitTiming=BeginTiming(publicationTimings,publicationTimingCursor,tick,pendingConsumerValue);
        LogEvent(frame,"physics.publication","scene=%llu slot=%u backpressure=1 consumer_completed=%llu wait_value=%llu gpu_wait_timed=%u",static_cast<unsigned long long>(handle),chosenIndex,static_cast<unsigned long long>(completed),static_cast<unsigned long long>(pendingConsumerValue),unsigned(waitTiming!=nullptr));
    }
    s.consumer.Wait(stream,pendingConsumerValue);
    if(waitTiming)Cuda(cudaEventRecord(waitTiming->end,stream),"end GPU publication consumer wait");
    // The legacy mesh ABI still reads the combined physical/population range.
    // Copying into it does not alter object IDs, control words or shader indices.
    if(s.groupActive[0]){
        auto& group=Group(0,s.groupSlots[0]);
        if(group.instances.handle)Cuda(cudaMemcpyAsync(s.instances.pointer,group.instances.pointer,uint64_t(visualHost.size())*sizeof(GpuInstance),cudaMemcpyDeviceToDevice,stream),"merge physics visual partition");
        Cuda(cudaMemcpyAsync(s.control.pointer,group.control.pointer,48,cudaMemcpyDeviceToDevice,stream),"merge physics visual control");
    }else Cuda(cudaMemsetAsync(s.control.pointer,0,48,stream),"initialize empty GPU visual control");
    if(insectCapacity){
        auto& group=Group(1,s.groupSlots[1]);
        // Only the old combined ABI needs a rendezvous. The population group
        // has already been produced and signalled on its dedicated CUDA stream.
        Cuda(cudaStreamWaitEvent(stream,populationPackReady,0),"join population for legacy mesh assembly");
        Cuda(cudaMemcpyAsync(static_cast<GpuInstance*>(s.instances.pointer)+visualHost.size(),
            group.instances.pointer,
            (uint64_t(insectCount)+4*LAB_PHYSICS_POPULATION_DETAIL_MAX_V1+LAB_POPULATION_SHADOW_MAX_V1)*sizeof(GpuInstance),
            cudaMemcpyDeviceToDevice,stream),"merge independent population instances");
        PopulationComposeControl(stream,static_cast<const uint32_t*>(group.control.pointer),
            static_cast<uint32_t*>(s.control.pointer),uint32_t(visualHost.size()));
        Cuda(cudaGetLastError(),"merge independent population control");
        // DX12 may retire the private population group before this legacy CUDA
        // copy completes. Its next reuse must also wait for this event.
        Cuda(cudaEventRecord(populationCopiesDone[s.groupSlots[1]],stream),"retire population legacy copy");
        populationCopied[s.groupSlots[1]]=true;
    }
    // The combined producer retires when PhysX/population packing is queued.
    // Snow.Splat waits solely on the independent snow group's CUDA signal.
    s.producer.Signal(stream,s.value);
    s.submission->SignalEnqueued(s.value);
    pendingPublication=nullptr;
}
uint32_t PhysicsRuntime::Impl::Scene::Acquire(LabGpuPublicationV1* out){
    if(!ReservePublication(out,false))return 0;
    CompletePublication();return 1;
}
void PhysicsRuntime::Impl::Scene::Release(void* token,uint64_t generation){
    for(auto& s:slots)if(token==&s){
        if(generation!=s.leaseGeneration)return; // stale lease cannot free reused storage
        if(!s.leased)Fail("PhysX GPU publication released twice");
        for(uint32_t kind=0;kind<3;kind++)if(s.groupActive[kind]){
            auto& group=Group(kind,s.groupSlots[kind]);
            if(!group.leased||group.leaseGeneration!=generation)Fail("PhysX GPU group lease mismatch");
            group.leased=false;
        }
        s.leased=false;s.groupActive.fill(false);return;
    }
}
void PhysicsRuntime::Impl::Scene::QueueProbe(){
    if(probeHost.empty())return;auto& p=probeSlots[probeCursor];
    if(p.pending){auto ready=cudaEventQuery(p.done);if(ready==cudaErrorNotReady){LogEvent(frame,"physics.probe","busy=1");return;}Cuda(ready,"probe completion");}
    // Each probe ring slot has its own pinned destination, never reused while
    // its async copy is in flight. No all-body host pose copy exists.
    Cuda(cudaEventRecord(p.gpuBegin,stream),"begin physics GPU timing");Driver(cuEventRecord(uploadReady,reinterpret_cast<CUstream>(stream)),"stage contact query");
    Gpu(scene->getDirectGPUAPI().copyContactData(contacts,contactCount,ContactCapacity,uploadReady,contactReady),"PhysX GPU contact query failed");Driver(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream),contactReady,0),"wait GPU contacts");
    PhysicsPackProbes(stream,probeDevice,uint32_t(probeHost.size()),current,velocity,angular,contacts,contactCount,ContactCapacity,actorDevice,uint32_t(actorHost.size()),probeOutput);Cuda(cudaGetLastError(),"compact GPU contact probes");
    Cuda(cudaMemcpyAsync(p.host,probeOutput,probeHost.size()*sizeof(LabPhysicsStateV1),cudaMemcpyDeviceToHost,stream),"copy compact behavior/audio probes");Cuda(cudaMemcpyAsync(p.overflow,contactCount,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream),"copy contact count");Cuda(cudaEventRecord(p.gpuEnd,stream),"end physics GPU timing");Cuda(cudaEventRecord(p.done,stream),"publish probe completion");p.pending=true;p.tick=tick;p.frame=frame;p.seconds=simulationTime;p.count=uint32_t(probeHost.size());probeCursor=(probeCursor+1)%uint32_t(probeSlots.size());
}
uint32_t PhysicsRuntime::Impl::Scene::Poll(LabPhysicsSnapshotV1* out){
    if(!out||out->structSize<sizeof(*out)||!out->states)return 0;PxScopedCudaLock lock(*runtime.cuda);ProbeSlot* newest{};
    for(auto& p:probeSlots){if(!p.pending||p.frame<=lastProbeFrame||p.count>out->stateCapacity)continue;auto ready=cudaEventQuery(p.done);if(ready==cudaErrorNotReady)continue;Cuda(ready,"query asynchronous physical state");if(!newest||p.frame>newest->frame)newest=&p;}
    if(!newest)return 0;auto& p=*newest;std::copy_n(p.host,p.count,out->states);out->stateCount=p.count;out->simulationTick=p.tick;out->sourceFrame=p.frame;out->simulationSeconds=p.seconds;out->contactOverflow=*p.overflow>=ContactCapacity;lastProbeFrame=p.frame;
    float ms{};Cuda(cudaEventElapsedTime(&ms,p.gpuBegin,p.gpuEnd),"measure physics probe GPU cost");LogEvent(frame,"physics.probe","source_frame=%llu gpu_ms=%.4f states=%u bytes=%u contact_pairs=%u overflow=%u",static_cast<unsigned long long>(p.frame),ms,p.count,uint32_t(p.count*sizeof(LabPhysicsStateV1)+4),*p.overflow,out->contactOverflow);
    if(out->contactOverflow)Fail("PhysX GPU contact query capacity exceeded; scene must be resized");return 1;
}

void PhysicsRuntime::Impl::Scene::CaptureEditPoses(){
    if(!started)return;
    Cuda(cudaStreamSynchronize(stream),"retire consumers before editor capture");
    if(topologyDirty)Fail("PhysX: editor capture requires committed topology");
    ReadState(true);
    if(dynamicCount)Cuda(cudaMemcpyAsync(editPoseHost,current,dynamicCount*sizeof(PxTransform),cudaMemcpyDeviceToHost,stream),"bounded editor body poses");
    Cuda(cudaStreamSynchronize(stream),"complete explicit editor snapshot");
    for(auto& b:bodies)if(b.actor&&b.index!=PhysicsInvalidIndex){
        if(b.index>=dynamicCount||!editPoseHost[b.index].isSane())Fail("PhysX: invalid editor GPU pose mapping");
        b.initial=editPoseHost[b.index].getNormalized();b.committed=b.initial;
    }
}
void PhysicsRuntime::Impl::Scene::WriteEditPoses(){
    // Paused publication is an authored preview, not a mutation of PhysX contacts,
    // joint shaders, broadphase data, or CPU shadows of GPU-driven actors.
    Cuda(cudaStreamSynchronize(stream),"retire paused preview consumers");
    for(auto& b:bodies)if(b.actor&&b.index!=PhysicsInvalidIndex){
        if(b.index>=dynamicCount)Fail("PhysX: invalid editor preview index");editPoseHost[b.index]=b.initial;
    }
    if(started&&dynamicCount){
        Cuda(cudaMemcpyAsync(current,editPoseHost,dynamicCount*sizeof(PxTransform),cudaMemcpyHostToDevice,stream),"publish authored preview poses");
        Cuda(cudaMemcpyAsync(previous,current,dynamicCount*sizeof(PxTransform),cudaMemcpyDeviceToDevice,stream),"reset editor preview interpolation");
    }
    commands.clear();accumulator=0;insectAccumulator=0;alpha=0;descriptorsDirty=true;
    if(started)RefreshDescriptors();
    Cuda(cudaStreamSynchronize(stream),"complete authored preview publication");
}
void PhysicsRuntime::Impl::Scene::CommitEditPoses(){
    Cuda(cudaStreamSynchronize(stream),"retire editor preview before commit");
    uint32_t count=0;
    // Stage only changed dynamic actors. GPU indices are actor identities, not
    // vector slots; never teleport unselected hands, the fan, or other bodies.
    for(auto& b:bodies)if(b.actor&&PoseChanged(b.initial,b.committed)){
        auto* dynamic=b.actor->is<PxRigidDynamic>();
        if(dynamic&&started){
            auto index=dynamic->getGPUIndex();
            if(b.index==PhysicsInvalidIndex||b.index>=indexHost.size()||index==0xffffffffu||index!=indexHost[b.index])Fail("PhysX: editor GPU identity changed during pause");
            for(uint32_t i=0;i<count;i++)if(editIndexHost[i]==index)Fail("PhysX: duplicate editor GPU index");
            editPoseHost[count]=b.initial;editIndexHost[count++]=index;
        }
    }
    if(count){
        Cuda(cudaMemcpyAsync(editPoseDevice,editPoseHost,count*sizeof(PxTransform),cudaMemcpyHostToDevice,stream),"stage grouped authored poses");
        Cuda(cudaMemcpyAsync(editIndexDevice,editIndexHost,count*sizeof(PxRigidDynamicGPUIndex),cudaMemcpyHostToDevice,stream),"stage grouped authored identities");
        Driver(cuEventRecord(uploadReady,reinterpret_cast<CUstream>(stream)),"publish isolated edit staging");auto& api=scene->getDirectGPUAPI();
        Gpu(api.setRigidDynamicData(editPoseDevice,editIndexDevice,PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE,count,uploadReady,poseReady),"PhysX GPU authoring poses");
        Gpu(api.setRigidDynamicData(editZeroDevice,editIndexDevice,PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY,count,poseReady,velocityReady),"PhysX GPU clear authored velocity");
        Gpu(api.setRigidDynamicData(editZeroDevice,editIndexDevice,PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY,count,velocityReady,stateReady),"PhysX GPU clear authored angular velocity");
        Driver(cuStreamWaitEvent(reinterpret_cast<CUstream>(stream),stateReady,0),"wait grouped editor commit");
        Cuda(cudaStreamSynchronize(stream),"complete grouped editor commit before CPU scene updates");
    }
    // Static setters remain supported with Direct GPU API; runtime dynamic pose
    // setters do not. Apply statics once, then rebuild D6 local frames against
    // the new static poses (the extension uses static inverse poses in c2b).
    for(auto& b:bodies)if(b.actor&&PoseChanged(b.initial,b.committed)){
        if(auto* fixed=b.actor->is<PxRigidStatic>())fixed->setGlobalPose(b.initial,false);
        else if(!started)static_cast<PxRigidDynamic*>(b.actor)->setGlobalPose(b.initial,false);
        if(b.kinematic){static_cast<PxRigidDynamic*>(b.actor)->setKinematicTarget(b.initial);b.lastTarget=b.initial;}
        b.committed=b.initial;b.geometry.pose={{b.initial.p.x,b.initial.p.y,b.initial.p.z},{b.initial.q.x,b.initial.q.y,b.initial.q.z,b.initial.q.w}};
    }
    for(auto& j:joints)if(j.joint){
        j.joint->setLocalPose(PxJointActorIndex::eACTOR0,P(j.description.frameA));j.joint->setLocalPose(PxJointActorIndex::eACTOR1,P(j.description.frameB));
        j.committedFrameA=j.description.frameA;j.committedFrameB=j.description.frameB;
    }
    runtime.Check();
    if(populationBodyPoses)Cuda(cudaMemsetAsync(populationBodyPoses,0,desc.maxBodies*sizeof(PopulationBodyPoseGpu),stream),"invalidate committed collider sweep history");
    descriptorsDirty=true;if(started){ReadState(true);RefreshDescriptors();}
    Cuda(cudaStreamSynchronize(stream),"complete authored state before simulate");
    for(auto& p:probeSlots)p.pending=false;for(auto& p:populationReadbacks)p.pending=false;
}
uint32_t PhysicsRuntime::Impl::Scene::Pause(bool value){
    PxScopedCudaLock lock(*runtime.cuda);if(value==paused)return 1;
    if(value){if(started&&topologyDirty)return 0;CaptureEditPoses();paused=true;commands.clear();accumulator=0;alpha=0;edited=false;}
    else{
        if(edited){LabPhysicsEditValidationV1 v{sizeof(v)};if(!ValidateEdit(&v))return 0;CommitEditPoses();}
        else Cuda(cudaStreamSynchronize(stream),"retire pause publications before resume");
        paused=false;lastTime=0;commands.clear();accumulator=0;insectAccumulator=0;edited=false;
    }
    return 1;
}
uint32_t PhysicsRuntime::Impl::Scene::EditSnapshot(LabPhysicsEditSnapshotV1* out){
    if(!out||out->structSize<sizeof(*out)||!out->bodies||!out->joints)return 0;
    uint32_t nb=0,nj=0;for(auto& b:bodies)nb+=b.actor!=nullptr;for(auto& j:joints)nj+=j.joint!=nullptr;
    if(nb>out->bodyCapacity||nj>out->jointCapacity||(!paused&&topologyDirty))return 0;
    PxScopedCudaLock lock(*runtime.cuda);if(!paused)CaptureEditPoses();nb=nj=0;
    for(auto& b:bodies)if(b.actor){const auto& t=b.initial;out->bodies[nb++]={b.handle,{{t.p.x,t.p.y,t.p.z},{t.q.x,t.q.y,t.q.z,t.q.w}}};}
    for(auto& j:joints)if(j.joint)out->joints[nj++]={j.handle,j.description.frameA,j.description.frameB};
    out->bodyCount=nb;out->jointCount=nj;out->simulationSeconds=simulationTime;return 1;
}
uint32_t PhysicsRuntime::Impl::Scene::Edit(const LabPhysicsEditBatchV1& batch){
    if(!paused||batch.structSize<sizeof(batch)||batch.bodyCount>bodies.size()||batch.jointCount>joints.size()||(batch.bodyCount&&!batch.bodies)||(batch.jointCount&&!batch.joints))return 0;
    for(uint32_t i=0;i<batch.bodyCount;i++){const auto& e=batch.bodies[i];if(!FindBody(e.body)||!Valid(e.pose)||V(e.pose.position).magnitude()>64)return 0;for(uint32_t j=0;j<i;j++)if(batch.bodies[j].body==e.body)return 0;}
    for(uint32_t i=0;i<batch.jointCount;i++){const auto& e=batch.joints[i];if(!FindJoint(e.joint)||!Valid(e.frameA)||!Valid(e.frameB)||V(e.frameA.position).magnitude()>256||V(e.frameB.position).magnitude()>256)return 0;for(uint32_t j=0;j<i;j++)if(batch.joints[j].joint==e.joint)return 0;}
    PxScopedCudaLock lock(*runtime.cuda);
    std::vector<PxTransform> oldBodies;std::vector<LabPhysicsEditJointV1> oldJoints;
    oldBodies.reserve(batch.bodyCount);oldJoints.reserve(batch.jointCount);
    for(uint32_t i=0;i<batch.bodyCount;i++){auto* b=FindBody(batch.bodies[i].body);oldBodies.push_back(b->initial);b->initial=P(batch.bodies[i].pose).getNormalized();}
    for(uint32_t i=0;i<batch.jointCount;i++){const auto& e=batch.joints[i];auto* j=FindJoint(e.joint);oldJoints.push_back({e.joint,j->description.frameA,j->description.frameB});j->description.frameA=e.frameA;j->description.frameB=e.frameB;}
    LabPhysicsEditValidationV1 result{sizeof(result)};
    if(!ValidateEdit(&result)){
        for(uint32_t i=0;i<batch.bodyCount;i++)FindBody(batch.bodies[i].body)->initial=oldBodies[i];
        for(const auto& e:oldJoints){auto* j=FindJoint(e.joint);j->description.frameA=e.frameA;j->description.frameB=e.frameB;}
        return 0;
    }
    edited=false;for(const auto& b:bodies)if(b.actor&&PoseChanged(b.initial,b.committed))edited=true;
    for(const auto& j:joints)if(j.joint&&(PoseChanged(P(j.description.frameA),P(j.committedFrameA))||PoseChanged(P(j.description.frameB),P(j.committedFrameB))))edited=true;
    WriteEditPoses();return 1;
}
uint32_t PhysicsRuntime::Impl::Scene::Translate(const LabPhysicsBodyV1* selected,uint32_t n,LabVec3V1 offset){
    if(!paused||!n||!selected||n>bodies.size()||!Valid(offset)||V(offset).magnitude()>.25f)return 0;
    std::vector<LabPhysicsEditBodyV1> edits;std::vector<LabPhysicsEditJointV1> anchors;
    auto contains=[&](LabPhysicsBodyV1 h){return h&&std::find(selected,selected+n,h)!=selected+n;};
    for(uint32_t i=0;i<n;i++){auto* b=FindBody(selected[i]);if(!b)return 0;auto t=b->initial;t.p+=V(offset);edits.push_back({b->handle,{{t.p.x,t.p.y,t.p.z},{t.q.x,t.q.y,t.q.z,t.q.w}}});}
    for(auto& j:joints)if(j.joint){const auto& d=j.description;bool a=contains(d.bodyA),b=contains(d.bodyB);if(a==b)continue;
        // Keep the connection's authored rest relation when one endpoint is
        // translated. World anchors and the other local frame move coherently.
        auto e=LabPhysicsEditJointV1{j.handle,d.frameA,d.frameB};auto other=a?FindBody(d.bodyB):FindBody(d.bodyA);
        auto& local=a?e.frameB:e.frameA;PxTransform parent=other?other->initial:PxTransform(PxIdentity),world=parent*P(local);world.p+=V(offset);world=parent.getInverse()*world;
        local={{world.p.x,world.p.y,world.p.z},{world.q.x,world.q.y,world.q.z,world.q.w}};anchors.push_back(e);
    }
    LabPhysicsEditBatchV1 batch{sizeof(batch),uint32_t(edits.size()),edits.data(),uint32_t(anchors.size()),0,anchors.data()};return Edit(batch);
}
uint32_t PhysicsRuntime::Impl::Scene::ValidateEdit(LabPhysicsEditValidationV1* out){
    if(!out||out->structSize<sizeof(*out))return 0;*out={sizeof(*out),0};
    if(!paused)return 0;out->valid=1;
    for(const auto& b:bodies)if(b.actor&&(!b.initial.isSane()||!b.initial.p.isFinite()||b.initial.p.magnitude()>64)){out->valid=0;out->bodyA=b.handle;return 0;}
    for(const auto& j:joints)if(j.joint){
        const auto& d=j.description;auto* a=d.bodyA?FindBody(d.bodyA):nullptr;auto* b=FindBody(d.bodyB);
        if((d.bodyA&&!a)||!b||!Valid(d.frameA)||!Valid(d.frameB)){out->valid=0;return 0;}
        auto wa=(a?a->initial:PxTransform(PxIdentity))*P(d.frameA),wb=b->initial*P(d.frameB),relative=wa.getInverse()*wb;
        const float axis[]{relative.p.x,relative.p.y,relative.p.z};float error=0;
        for(uint32_t i=0;i<3;i++){
            if(d.motion[i]==LAB_PHYSICS_LOCKED)error=std::max(error,PxAbs(axis[i]));
            else if(d.motion[i]==LAB_PHYSICS_LIMITED)error=std::max(error,std::max(d.linearLower[i]-axis[i],axis[i]-d.linearUpper[i]));
        }
        auto q=relative.q;if(q.w<0)q=-q;PxQuat twist(q.x,0,0,q.w);float n=twist.magnitudeSquared();
        twist=n>1e-12f?twist.getNormalized():PxQuat(PxIdentity);auto swing=q*twist.getConjugate();
        float angular[]{2*std::atan2(twist.x,twist.w),2*std::atan2(swing.y,swing.w),2*std::atan2(swing.z,swing.w)};
        float angularError=0;for(uint32_t i=0;i<3;i++){
            if(d.motion[i+3]==LAB_PHYSICS_LOCKED)angularError=std::max(angularError,PxAbs(angular[i]));
            else if(d.motion[i+3]==LAB_PHYSICS_LIMITED){
                float low=i==0?d.twistLower:-(i==1?d.swingY:d.swingZ),high=i==0?d.twistUpper:(i==1?d.swingY:d.swingZ);
                angularError=std::max(angularError,std::max(low-angular[i],angular[i]-high));
            }
        }
        if(error>.025f||angularError>.15f){out->valid=0;out->bodyA=d.bodyA;out->bodyB=d.bodyB;out->penetration=error;return 0;}
    }
    for(size_t i=0;i<bodies.size();i++)for(size_t k=i+1;k<bodies.size();k++){
        auto& a=bodies[i];auto& b=bodies[k];if(!a.actor||!b.actor||!a.collisionEnabled||!b.collisionEnabled)continue;
        // Existing static/static intersections are not PhysX contacts and
        // must not block an unrelated edit. Still validate moved statics.
        if(a.actor->is<PxRigidStatic>()&&b.actor->is<PxRigidStatic>()&&!PoseChanged(a.initial,a.committed)&&!PoseChanged(b.initial,b.committed))continue;
        if(!(a.geometry.layer&b.collisionMask)||!(b.geometry.layer&a.collisionMask)||(a.geometry.collisionGroup&&a.geometry.collisionGroup==b.geometry.collisionGroup))continue;
        bool connected=false;for(auto& j:joints)if(j.joint&&!j.description.collideConnected&&((j.description.bodyA==a.handle&&j.description.bodyB==b.handle)||(j.description.bodyA==b.handle&&j.description.bodyB==a.handle))){connected=true;break;}if(connected)continue;
        PxShape *sa{},*sb{};if(a.actor->getShapes(&sa,1)!=1||b.actor->getShapes(&sb,1)!=1)continue;PxVec3 direction;PxReal depth;
        if(PxGeometryQuery::computePenetration(direction,depth,sa->getGeometry(),a.initial*sa->getLocalPose(),sb->getGeometry(),b.initial*sb->getLocalPose())&&depth>.025f&&depth>out->penetration){out->valid=0;out->bodyA=a.handle;out->bodyB=b.handle;out->penetration=depth;}
    }
    return out->valid;
}

PhysicsRuntime::PhysicsRuntime():impl_(std::make_unique<Impl>()){}
PhysicsRuntime::~PhysicsRuntime(){Shutdown();}
void PhysicsRuntime::Initialize(ID3D12Device* device){
    auto& r=*impl_;if(r.physics||!device)Fail("PhysX initialization order: DX12 device required exactly once");Driver(cuInit(0),"initialize CUDA driver");auto luid=device->GetAdapterLuid();int count{};Driver(cuDeviceGetCount(&count),"enumerate CUDA devices");int selected=-1;
    for(int i=0;i<count;i++){char identity[8]{};unsigned int nodeMask{};Driver(cuDeviceGetLuid(identity,&nodeMask,i),"read CUDA adapter LUID");if(std::memcmp(identity,&luid,8)==0&&nodeMask){selected=i;break;}}
    if(selected<0)Fail("PhysX: no CUDA adapter matches the OpenXR DX12 adapter LUID");int major{},minor{};Driver(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,selected),"query CUDA architecture");Driver(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,selected),"query CUDA architecture");if(major!=12||minor!=0)Fail("PhysX reference build requires SM120 on the DX12/CUDA adapter");
    r.foundation=PxCreateFoundation(PX_PHYSICS_VERSION,r.allocator,r.errors);if(!r.foundation)Fail("PhysX foundation creation failed");r.physics=PxCreatePhysics(PX_PHYSICS_VERSION,*r.foundation,PxTolerancesScale(),false,nullptr);if(!r.physics)Fail("PhysX SDK creation failed");r.extensions=PxInitExtensions(*r.physics,nullptr);if(!r.extensions)Fail("PhysX extensions initialization failed");
    PxCudaContextManagerDesc d;d.deviceOrdinal=selected;r.cuda=PxCreateCudaContextManager(*r.foundation,d,PxGetProfilerCallback());if(!r.cuda)Fail("PhysX CUDA context manager creation failed");r.Check();if(r.cuda->getDevice()!=selected)Fail("PhysX CUDA context is on the wrong DX12 adapter");
    r.dispatcher=PxDefaultCpuDispatcherCreate(2);if(!r.dispatcher)Fail("PhysX CPU orchestration dispatcher creation failed");r.Check();LogEvent(0,"physics.initialize","physx=%u cuda_device=%d luid=%08x:%08x sm=%d%d workers=2",PX_PHYSICS_VERSION,selected,unsigned(luid.HighPart),unsigned(luid.LowPart),major,minor);
}
void PhysicsRuntime::Shutdown(){auto& r=*impl_;for(auto& [id,o]:r.owners)for(auto& [h,s]:o->scenes)s->Destroy();r.owners.clear();if(r.dispatcher)r.dispatcher->release();r.dispatcher=nullptr;if(r.cuda)r.cuda->release();r.cuda=nullptr;if(r.extensions)PxCloseExtensions();r.extensions=false;if(r.physics)r.physics->release();r.physics=nullptr;if(r.foundation)r.foundation->release();r.foundation=nullptr;}
uint64_t PhysicsRuntime::CreateOwner(const LabGpuInteropServiceV1& interop){
    auto& r=*impl_;if(!r.physics||interop.version!=1||interop.structSize<sizeof(interop))Fail("PhysX owner requires initialized Core and gpu_interop_v1");auto o=std::make_unique<Impl::Owner>();o->runtime=&r;o->id=r.nextOwner++;o->interop=interop;o->service={1,sizeof(LabPhysicsServiceV1),o.get()};
    auto& t=o->service;
    t.CreateScene=[](void* c,const LabPhysicsSceneDescV1* d,LabPhysicsSceneV1* out)->LabU32{if(out)*out=0;if(!c||!d||!out||d->structSize<sizeof(*d)||!Valid(d->gravity)||d->maxBodies<1||d->maxBodies>4096||!Finite(d->fixedStepSeconds)||d->fixedStepSeconds<1.f/1000||d->fixedStepSeconds>1.f/60||d->maxSubsteps<1||d->maxSubsteps>16)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto h=uint64_t(o.runtime->Generation())<<32|1;auto s=std::make_unique<Impl::Scene>(*o.runtime,o,h,*d);s->Initialize();o.scenes.emplace(h,std::move(s));*out=h;return 1;};
    t.DestroyScene=[](void* c,LabPhysicsSceneV1 h)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto i=o.scenes.find(h);if(i==o.scenes.end())return 0;i->second->Destroy();o.scenes.erase(i);return 1;};
    t.CreateBody=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsBodyDescV1* d,LabPhysicsBodyV1* out)->LabU32{if(!c||!d)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->CreateBody(*d,out):0;};
    t.DestroyBody=[](void* c,LabPhysicsSceneV1 h,LabPhysicsBodyV1 b)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->DestroyBody(b):0;};
    t.CreateD6=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsD6DescV1* d,LabPhysicsJointV1* out)->LabU32{if(!c||!d)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->CreateD6(*d,out):0;};
    t.ConfigureD6=[](void* c,LabPhysicsSceneV1 h,LabPhysicsJointV1 j,const LabPhysicsD6DescV1* d)->LabU32{if(!c||!d)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->ConfigureD6(j,*d):0;};
    t.DestroyJoint=[](void* c,LabPhysicsSceneV1 h,LabPhysicsJointV1 j)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->DestroyJoint(j):0;};
    t.SetMotors=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsMotorV1* m,LabU32 n)->LabU32{if(!c||(n&&!m))return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s||n>s->joints.size())return 0;for(uint32_t i=0;i<n;i++)if(m[i].structSize<sizeof(*m)||!s->FindJoint(m[i].joint)||!Valid(m[i].target)||!Valid(m[i].linearVelocity)||!Valid(m[i].angularVelocity)||!Finite(m[i].frequencyHz)||m[i].frequencyHz<0||m[i].frequencyHz>1.f/(8*s->desc.fixedStepSeconds)||!Finite(m[i].phaseRadians)||!Finite(m[i].amplitudeRadians)||std::abs(m[i].amplitudeRadians)>1.5f)return 0;for(uint32_t i=0;i<n;i++)s->FindJoint(m[i].joint)->motor=m[i];return 1;};
    t.SubmitCommands=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsCommandV1* commands,LabU32 n)->LabU32{if(!c||(n&&!commands))return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s||n>s->desc.maxBodies*4||s->commands.size()+n>s->desc.maxBodies*8)return 0;for(uint32_t i=0;i<n;i++){auto& v=commands[i];auto* b=s->FindBody(v.body);if(v.structSize<sizeof(v)||!b||v.kind>LAB_PHYSICS_KINEMATIC_TARGET||!Valid(v.value)||!Valid(v.target)||!b->actor->is<PxRigidDynamic>()||((v.kind==LAB_PHYSICS_KINEMATIC_TARGET)!=b->kinematic))return 0;}for(uint32_t i=0;i<n;i++){const auto& v=commands[i];if(v.kind!=LAB_PHYSICS_IMPULSE&&v.kind!=LAB_PHYSICS_ANGULAR_IMPULSE)std::erase_if(s->commands,[&](const auto& old){return old.body==v.body&&old.kind==v.kind;});s->commands.push_back(v);}return 1;};
    t.SetAeroSurfaces=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsAeroSurfaceV1* a,LabU32 n)->LabU32{if(!c||(n&&!a))return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s||n>s->desc.maxBodies)return 0;for(uint32_t i=0;i<n;i++){auto* wing=s->FindBody(a[i].wing);auto* carrier=s->FindBody(a[i].carrier);if(a[i].structSize<sizeof(*a)||!wing||!carrier||wing->kinematic||carrier->kinematic||!wing->actor->is<PxRigidDynamic>()||!carrier->actor->is<PxRigidDynamic>()||!Valid(a[i].liftAxis)||!Finite(a[i].liftCoefficient)||!Finite(a[i].dragCoefficient)||!Finite(a[i].maxLift)||!Finite(a[i].maxDrag)||a[i].liftCoefficient<0||a[i].dragCoefficient<0||a[i].maxLift<0||a[i].maxDrag<0)return 0;}s->aero.clear();if(n)s->aero.assign(a,a+n);s->descriptorsDirty=true;return 1;};
    t.SetVisuals=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsVisualV1* v,LabU32 n,const LabPhysicsLightV1* l,LabU32 nl)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->SetVisuals(v,n,l,nl):0;};
    t.GetVisualProvider=[](void* c,LabPhysicsSceneV1 h,LabGpuPublicationProviderV1* out)->LabU32{if(!c||!out||out->structSize<sizeof(*out))return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s)return 0;*out=s->provider;return 1;};
    t.SetVisible=[](void* c,LabPhysicsSceneV1 h,LabU32 visible)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s)return 0;s->visible=visible!=0;return 1;};
    t.SetLightPower=[](void* c,LabPhysicsSceneV1 h,LabU32 lightIndex,float power)->LabU32{if(!c||!Finite(power)||power<0)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s)return 0;for(auto& l:s->lights)if(l.sceneLightIndex==lightIndex){l.power=power;s->descriptorsDirty=true;return 1;}return 0;};
    t.SetProbes=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsBodyV1* b,LabU32 n)->LabU32{if(!c||(n&&!b)||n>LABOVR_PHYSICS_MAX_PROBES_V1)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s)return 0;for(uint32_t i=0;i<n;i++)if(!s->FindBody(b[i]))return 0;s->lastProbeFrame=s->frame;s->probes.clear();if(n)s->probes.assign(b,b+n);s->descriptorsDirty=true;return 1;};
    t.PollSnapshot=[](void* c,LabPhysicsSceneV1 h,LabPhysicsSnapshotV1* out)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->Poll(out):0;};
    t.SetBodyCollisionEnabled=[](void* c,LabPhysicsSceneV1 h,LabPhysicsBodyV1 b,LabU32 enabled)->LabU32{if(!c||enabled>1)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->SetBodyCollisionEnabled(b,enabled!=0):0;};
    t.SetBodySpeculativeCCD=[](void* c,LabPhysicsSceneV1 h,LabPhysicsBodyV1 b,LabU32 enabled)->LabU32{if(!c||enabled>1)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->SetBodySpeculativeCCD(b,enabled!=0):0;};
    t.CreatePopulation=[](void* c,LabPhysicsSceneV1 h,const LabPopulationDescV1* d)->LabU32{if(!c||!d)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->ConfigurePopulation(*d):0;};
    t.CreateSnowfall=[](void* c,LabPhysicsSceneV1 h,LabU32 seed)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->ConfigureSnowfall(seed):0;};
    t.SetPopulationCount=[](void* c,LabPhysicsSceneV1 h,LabU32 n)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->ResizePopulation(n):0;};
    t.SetFlightSettings=[](void* c,LabPhysicsSceneV1 h,const LabFlightSettingsV1* f)->LabU32{if(!c||!f||!ValidFlight(*f))return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s||!s->insectCapacity||f->seed!=s->population.flight.seed)return 0;s->population.flight=*f;s->populationEffects.fleeSeconds=f->impactSeconds;return 1;};
    t.SetPopulationInput=[](void* c,LabPhysicsSceneV1 h,const LabPopulationInputV1* in)->LabU32{if(!c||!in||in->structSize<sizeof(*in)||!Valid(in->playerPosition)||!Valid(in->aimOrigin)||!Valid(in->aimDirection))return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s||!s->insectCapacity)return 0;for(auto hand:in->hands)if(hand&&!s->FindBody(hand))return 0;if(in->hands[0]!=s->populationInput.hands[0]||in->hands[1]!=s->populationInput.hands[1])s->descriptorsDirty=true;s->populationInput=*in;return 1;};
    t.SetPopulationEffects=[](void* c,LabPhysicsSceneV1 h,const LabPopulationEffectsV1* e)->LabU32{
        if(!c||!e||e->structSize<sizeof(*e))return 0;
        const float v[]{e->sweepRadius,e->impulse,e->fleeSeconds,e->avoidanceRadius,e->residualAversion,e->shadowDensity,e->shadowIntensity};
        for(float x:v)if(!Finite(x)||x<0)return 0;
        if(e->sweepRadius<.02f||e->sweepRadius>1||e->impulse>8||e->fleeSeconds>120||e->avoidanceRadius<.1f||e->avoidanceRadius>4||e->residualAversion>1||e->shadowDensity>1||e->shadowIntensity>1)return 0;
        auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s||!s->insectCapacity)return 0;s->populationEffects=*e;s->population.flight.impactSeconds=e->fleeSeconds;return 1;
    };
    t.SetPopulationHands=[](void* c,LabPhysicsSceneV1 h,const LabPopulationHandsV1* in)->LabU32{
        if(!c||!in||in->structSize<sizeof(*in))return 0;
        for(auto& v:in->hand)if(!Valid(v.position)||!Valid(v.previous)||!Valid(v.velocity)||V(v.velocity).magnitudeSquared()>400||(V(v.position)-V(v.previous)).magnitudeSquared()>.25f)return 0;
        auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s||!s->insectCapacity)return 0;s->populationHands=*in;return 1;
    };
    t.PollPopulation=[](void* c,LabPhysicsSceneV1 h,LabPopulationSnapshotV1* out)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->PollPopulation(out):0;};
    t.SetAirflows=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsAirflowV1* fields,LabU32 n)->LabU32{
        if(!c||n>8||(n&&!fields))return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s)return 0;
        for(uint32_t i=0;i<n;i++){const auto& f=fields[i];auto* b=s->FindBody(f.emitter);float norm=V(f.localDirection).magnitudeSquared();
            if(f.structSize<sizeof(f)||f.enabled>1||!b||!(b->actor->is<PxRigidDynamic>()||(f.referenceRpm==0.f&&b->actor->is<PxRigidStatic>()))||!Valid(f.localDirection)||norm<.99f||norm>1.01f||!Finite(f.radius)||f.radius<.05f||f.radius>4||!Finite(f.length)||f.length<.1f||f.length>16||!Finite(f.speedAtReferenceRpm)||f.speedAtReferenceRpm<0||f.speedAtReferenceRpm>80||!Finite(f.referenceRpm)||f.referenceRpm<0||!Finite(f.turbulence)||f.turbulence<0||f.turbulence>1)return 0;
        }
        if(n==s->airflows.size()&&(!n||!memcmp(fields,s->airflows.data(),n*sizeof(*fields))))return 1;
        {PxScopedCudaLock lock(*o.runtime->cuda);Cuda(cudaStreamSynchronize(s->stream),"retire airflow descriptor edits");}
        s->airflows.clear();if(n)s->airflows.assign(fields,fields+n);s->descriptorsDirty=true;return 1;
    };
    t.SetPopulationSpace=[](void* c,LabPhysicsSceneV1 h,const LabPopulationSpaceV1* v)->LabU32{
        if(!c||!v||v->structSize<sizeof(*v)||v->enabled>1||!Valid(v->center)||!Valid(v->halfExtent)||v->halfExtent.x<.1f||v->halfExtent.y<.1f||v->halfExtent.z<.1f||v->halfExtent.x>64||v->halfExtent.y>64||v->halfExtent.z>64||!Finite(v->returnStrength)||v->returnStrength<0||v->returnStrength>16)return 0;
        auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);if(!s)return 0;s->populationSpace=*v;return 1;
    };
    t.SetPaused=[](void* c,LabPhysicsSceneV1 h,LabU32 paused)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s&&paused<=1?s->Pause(paused!=0):0;};
    t.GetEditSnapshot=[](void* c,LabPhysicsSceneV1 h,LabPhysicsEditSnapshotV1* out)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->EditSnapshot(out):0;};
    t.ApplyEdit=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsEditBatchV1* batch)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?batch?s->Edit(*batch):0:0;};
    t.TranslateBodies=[](void* c,LabPhysicsSceneV1 h,const LabPhysicsBodyV1* b,LabU32 n,LabVec3V1 delta)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->Translate(b,n,delta):0;};
    t.ValidateEdit=[](void* c,LabPhysicsSceneV1 h,LabPhysicsEditValidationV1* out)->LabU32{if(!c)return 0;auto& o=*static_cast<Impl::Owner*>(c);auto* s=o.runtime->Find(o,h);return s?s->ValidateEdit(out):0;};
    auto id=o->id;r.owners.emplace(id,std::move(o));return id;
}
void PhysicsRuntime::ResetOwner(uint64_t id){auto& r=*impl_;auto i=r.owners.find(id);if(i==r.owners.end())return;for(auto& [h,s]:i->second->scenes)s->Destroy();i->second->scenes.clear();}
LabPhysicsServiceV1 PhysicsRuntime::Service(uint64_t id){auto it=impl_->owners.find(id);if(it==impl_->owners.end())Fail("PhysX owner unavailable");return it->second->service;}
uint32_t PhysicsRuntime::ReservePublication(uint64_t id,const LabGpuPublicationProviderV1& provider,LabGpuPublicationV1* out){
    auto owner=impl_->owners.find(id);if(owner==impl_->owners.end())return 0;
    for(auto& [handle,scene]:owner->second->scenes)
        if(provider.callbackContext==scene.get()&&provider.Acquire==scene->provider.Acquire&&provider.Release==scene->provider.Release)
            return scene->ReservePublication(out,true);
    return 0;
}
uint32_t PhysicsRuntime::DescribePublicationGroups(uint64_t id,const LabGpuPublicationProviderV1& provider,
    const LabGpuPublicationV1& publication,PhysicsPublicationGroupSync* out,uint32_t capacity)const{
    if(!out||capacity<3||!publication.token)return 0;
    auto owner=impl_->owners.find(id);if(owner==impl_->owners.end())return 0;
    for(const auto& [sceneId,scene]:owner->second->scenes){
        if(provider.callbackContext!=scene.get()||provider.Acquire!=scene->provider.Acquire||provider.Release!=scene->provider.Release)continue;
        for(const auto& lease:scene->slots)if(publication.token==&lease){
            if(!lease.leased||lease.leaseGeneration!=publication.tokenGeneration)return 0;
            uint32_t count=0;
            for(uint32_t kind=0;kind<3;kind++)if(lease.groupActive[kind]){
                auto& group=scene->Group(kind,lease.groupSlots[kind]);
                if(!group.leased||group.leaseGeneration!=publication.tokenGeneration)Fail("PhysX GPU group lease invalid");
                out[count++]={kind+1,group.producer.handle,group.consumer.handle,group.value,group.value,
                    0,kind==0?lease.physicalCount:0,
                    kind==0?group.storageGeneration:0,
                    kind==0?lease.rayRevision:0,
                    kind==0?sceneId:0,group.submission};
            }
            return count;
        }
    }
    return 0;
}
std::shared_ptr<GpuProducerSubmission> PhysicsRuntime::DescribePublicationSubmission(
    uint64_t id,const LabGpuPublicationProviderV1& provider,
    const LabGpuPublicationV1& publication)const{
    if(!publication.token)return {};
    auto owner=impl_->owners.find(id);if(owner==impl_->owners.end())return {};
    for(const auto& [sceneId,scene]:owner->second->scenes){
        if(provider.callbackContext!=scene.get()||provider.Acquire!=scene->provider.Acquire||
            provider.Release!=scene->provider.Release)continue;
        for(const auto& lease:scene->slots)
            if(publication.token==&lease&&lease.leased&&
                lease.leaseGeneration==publication.tokenGeneration)return lease.submission;
    }
    return {};
}
void PhysicsRuntime::Advance(double now,uint64_t inputFrame){impl_->Check();for(auto& [id,o]:impl_->owners)for(auto& [h,s]:o->scenes){s->Advance(now,inputFrame);s->CompletePublication();}}
}
