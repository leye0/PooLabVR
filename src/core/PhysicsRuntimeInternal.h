#pragma once
#include "PhysicsRuntime.h"
#include "GpuPublication.h"
#include "PhysicsDiagnostics.h"
#include "PhysicsGpuInterop.h"
#include "PhysicsCuda.h"
#include "PopulationCuda.h"
#include "GpuParticleField.h"
#include "PopulationPolicy.h"
#include "Core.h"
#include "SessionLog.h"
#include "ConsoleSpec.h"
#include <cudamanager/PxCudaContextManager.h>
#include <extensions/PxDefaultCpuDispatcher.h>
#include <extensions/PxRigidBodyExt.h>
#include <extensions/PxD6Joint.h>
#include <cuda.h>
#include <d3d12.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <span>
#include <vector>

namespace lab {
using namespace physx;
namespace physics_detail {
constexpr PxU32 SolverPositionIterations=10;
constexpr PxU32 SolverVelocityIterations=3;
inline void Cuda(cudaError_t result,const char* operation){if(result!=cudaSuccess){char message[512];snprintf(message,sizeof(message),"PhysX CUDA: %s: %s",operation,cudaGetErrorString(result));Fail(message);}}
inline void Driver(CUresult result,const char* operation){if(result!=CUDA_SUCCESS){const char* error{};cuGetErrorString(result,&error);char message[512];snprintf(message,sizeof(message),"PhysX CUDA driver: %s: %s",operation,error?error:"unknown");Fail(message);}}
inline void Gpu(bool result,const char* operation){if(!result)Fail(operation);}
inline PxVec3 V(LabVec3V1 v){return {v.x,v.y,v.z};}
inline PxTransform P(LabPoseV1 p){return {V(p.position),{p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w}};}
inline bool Finite(float v){return std::isfinite(v);}
inline bool Valid(LabVec3V1 v){return Finite(v.x)&&Finite(v.y)&&Finite(v.z);}
inline bool Valid(LabPoseV1 p){return Valid(p.position)&&P(p).isSane();}
inline bool PoseChanged(const PxTransform& a,const PxTransform& b){
    // Compare components, including q/-q equivalence: dot(q,q) can round below
    // one even for identical poses and must not mark untouched actors as edited.
    auto q=b.q;if(a.q.dot(q)<0)q=-q;
    return (a.p-b.p).magnitudeSquared()>1e-14f||(a.q-q).magnitudeSquared()>1e-14f;
}
inline bool ValidFlight(const LabFlightSettingsV1& f){
    if(f.structSize<sizeof(f))return false;
    const float values[]{f.nominalSpeed,f.maxSpeed,f.responseSeconds,f.turnIntensity,f.turnRadius,f.lightAttraction,f.comfortDistance,f.anticipationDistance,f.wander,f.vertical,f.avoidanceFailure,f.impactSeconds,f.impactMultiplier,f.mass,f.wingArea,f.liftCoefficient,f.dragCoefficient,f.wingFrequency,f.bodyRadius,f.detailDistance,f.nearDistance};
    for(float v:values)if(!Finite(v)||v<0)return false;
    return f.mass>=1e-7f&&f.mass<=.01f&&f.wingArea>=1e-8f&&f.wingArea<=.01f&&f.liftCoefficient<=8&&f.dragCoefficient<=8&&f.wingFrequency<=500&&f.responseSeconds<=2&&f.turnIntensity<=8&&f.turnRadius<=4&&f.lightAttraction<=8&&f.comfortDistance<=2&&f.anticipationDistance<=4&&f.wander<=4&&f.vertical<=4&&f.impactMultiplier<=10&&f.bodyRadius<=.1f&&f.detailDistance<=8&&f.nearDistance<=8&&f.nominalSpeed>0&&f.maxSpeed>=f.nominalSpeed&&f.maxSpeed<=12&&f.responseSeconds>=.02f&&f.turnIntensity>0&&f.turnRadius>=.02f&&f.comfortDistance>=.02f&&f.anticipationDistance>=f.comfortDistance&&f.avoidanceFailure<=1&&f.impactSeconds<=120&&f.impactMultiplier>=1&&f.mass>0&&f.wingArea>0&&f.liftCoefficient>0&&f.wingFrequency>0&&f.bodyRadius>=.001f;
}
template<class T> void Allocate(T*& p,size_t n){Cuda(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"allocate scene buffer");}
template<class T> void Upload(cudaStream_t stream,T* device,const std::vector<T>& host){if(!host.empty())Cuda(cudaMemcpyAsync(device,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice,stream),"upload scene descriptors");}
inline PxFilterFlags Filter(PxFilterObjectAttributes,PxFilterData a,PxFilterObjectAttributes,PxFilterData b,PxPairFlags& flags,const void*,PxU32){
    if(!(a.word0&b.word1)||!(b.word0&a.word1)||(a.word2&&a.word2==b.word2))return PxFilterFlag::eSUPPRESS;
    flags=PxPairFlag::eCONTACT_DEFAULT|PxPairFlag::eNOTIFY_CONTACT_POINTS;return PxFilterFlag::eDEFAULT;
}
struct Errors final:PxErrorCallback {
    std::atomic<bool> failed{};
    void reportError(PxErrorCode::Enum code,const char* message,const char* file,int line)override{
        LogEvent(0,"physics.error","code=%u file=%s line=%d message=%s",unsigned(code),file,line,message);
        switch(code){
            case PxErrorCode::eNO_ERROR:
            case PxErrorCode::eDEBUG_INFO:
            case PxErrorCode::eDEBUG_WARNING:
            case PxErrorCode::ePERF_WARNING:return;
            default:failed=true;OutputDebugStringA(message);OutputDebugStringA("\n");break;
        }
    }
};

}
using namespace physics_detail;
struct PhysicsRuntime::Impl {
    struct Owner;
    PxDefaultAllocator allocator;Errors errors;
    PxFoundation* foundation{};PxPhysics* physics{};PxCudaContextManager* cuda{};PxDefaultCpuDispatcher* dispatcher{};bool extensions{};
    uint32_t nextGeneration{1};uint64_t nextOwner{1},nextLeaseGeneration{1};
    struct Body {LabPhysicsBodyV1 handle{};PxRigidActor* actor{};PxMaterial* material{};PxConvexMesh* convex{};PxTransform initial{PxIdentity},committed{PxIdentity};uint32_t index{PhysicsInvalidIndex},collisionMask{};LabPhysicsBodyDescV1 geometry{};bool kinematic{},collisionEnabled{true};PxTransform lastTarget{PxIdentity};};
    struct Joint {
        LabPhysicsJointV1 handle{};PxD6Joint* joint{};LabPhysicsMotorV1 motor{};
        LabPhysicsD6DescV1 description{};LabPoseV1 committedFrameA{},committedFrameB{};
        double motorPhase{};
        // PhysX joint drive targets persist until changed; a repeated setter
        // unnecessarily dirties the constraint on each 120 Hz substep.
        bool lastDriveValid{};
        PxTransform lastDrivePosition{PxIdentity};
        PxVec3 lastDriveLinear{},lastDriveAngular{};
    };
    struct Scene {
        Impl& runtime;Owner& owner;LabPhysicsSceneV1 handle{};LabPhysicsSceneDescV1 desc{};PxScene* scene{};
        cudaStream_t stream{},snowfallStream{},populationStream{};CUevent uploadReady{},poseReady{},velocityReady{},stateReady{},forceReady{},forceDone{},torqueDone{},contactReady{};
        PxRigidDynamicGPUIndex* indices{};PxTransform *previous{},*current{};PxVec3 *velocity{},*angular{},*forces{},*torques{};
        LabPopulationEffectsV1 populationEffects{sizeof(LabPopulationEffectsV1),0,.18f,1.f,12.f,1.3f,.55f,.7f,.35f};
        LabPopulationHandsV1 populationHands{sizeof(LabPopulationHandsV1)};
        LabPopulationDescV1 population{};LabPopulationInputV1 populationInput{sizeof(LabPopulationInputV1),1};
        PopulationStateGpu* insects{};PopulationBodyGpu* populationBodies{};PopulationBodyPoseGpu* populationBodyPoses{};uint32_t* populationGrid{};PopulationSummaryGpu* populationSummary{};
        std::vector<PopulationBodyGpu> populationBodyHost;
        uint32_t insectCount{},insectCapacity{},birthGeneration{};double insectAccumulator{},populationClockOrigin{};bool populationClockStarted{};
        struct PopulationReadback {LabPopulationSnapshotV1* host{};cudaEvent_t done{};bool pending{};uint64_t sequence{};};
        std::array<PopulationReadback,3> populationReadbacks;uint32_t populationReadCursor{};uint64_t populationReadSequence{},populationPolledSequence{};
        uint32_t ConfigurePopulation(const LabPopulationDescV1&);uint32_t ResizePopulation(uint32_t);uint32_t PollPopulation(LabPopulationSnapshotV1*);
        uint32_t ConfigureSnowfall(uint32_t seed);
        void QueueSnowfall();
        void StepPopulation();void QueuePopulationReadback();
        void QueuePopulationPublication();
        PhysicsAirflowGpu* airflowDevice{};PhysicsDragGpu* dragDevice{};
        std::vector<LabPhysicsAirflowV1> airflows;std::vector<PhysicsAirflowGpu> airflowHost;std::vector<PhysicsDragGpu> dragHost;
        LabPopulationSpaceV1 populationSpace{sizeof(LabPopulationSpaceV1),0,{},{1,1,1},1};
        PhysicsVisualGpu* visualDevice{};PhysicsLightGpu* lightDevice{};PhysicsAeroGpu* aeroDevice{};PhysicsProbeGpu* probeDevice{};PhysicsActorGpu* actorDevice{};
        PxGpuContactPair* contacts{};uint32_t* contactCount{};LabPhysicsStateV1* probeOutput{};
        std::vector<Body> bodies;std::vector<Joint> joints;std::vector<PxRigidDynamicGPUIndex> indexHost;
        PxVec3 *forceHost{},*torqueHost{};
        PxTransform *editPoseHost{},*editPoseDevice{};PxRigidDynamicGPUIndex *editIndexHost{},*editIndexDevice{};PxVec3* editZeroDevice{};
        std::vector<LabPhysicsCommandV1> commands;std::vector<LabPhysicsAeroSurfaceV1> aero;
        std::vector<LabPhysicsVisualV1> visuals;std::vector<LabPhysicsLightV1> lights;std::vector<LabPhysicsBodyV1> probes;
        std::vector<PhysicsVisualGpu> visualHost;std::vector<PhysicsLightGpu> lightHost;std::vector<PhysicsAeroGpu> aeroHost;std::vector<PhysicsProbeGpu> probeHost;std::vector<PhysicsActorGpu> actorHost;
        bool paused{},edited{};
        bool topologyDirty{true},descriptorsDirty{true},started{},visible{true};uint32_t dynamicCount{};
        double populationSeconds{},accumulator{},lastTime{},simulationTime{},droppedSeconds{};uint64_t tick{},frame{};float alpha{};
        static constexpr uint32_t ContactCapacity=1024,MaxVisuals=2048,MaxLights=32;
        struct PublicationSlot {
            // The legacy combined buffers remain the active render contract until
            // consumers can bind the independent groups in subsequent steps.
            SharedBuffer instances,control;SharedFence producer,consumer;
            struct ResourceGroup {
                SharedBuffer instances,control,lighting;SharedFence producer,consumer;
                std::vector<uint32_t> bindings;
                uint64_t value{},leaseGeneration{},storageGeneration{};bool leased{};
                std::shared_ptr<GpuProducerSubmission> submission;
            } physicsVisual,population,snow;
            std::vector<LabGpuResourceRefV1> refs;
            std::array<uint32_t,3> groupSlots{};
            std::array<uint64_t,3> groupPriorConsumerValues{};
            std::array<bool,3> groupActive{};
            uint64_t value{},leaseGeneration{},rayRevision{};
            std::shared_ptr<GpuProducerSubmission> submission;
            uint32_t physicalCount{};
            bool leased{};
        };
        // Three persistent DX12/CUDA buffers are both the simulation state
        // and the rendered publication. A GPU snowfall kernel reads the prior
        // state and writes the next free publication: no 64 MB D2D snapshot.
        static constexpr uint32_t SnowfallRingSize=3;
        std::array<PublicationSlot,SnowfallRingSize> slots;uint32_t publicationCursor{};
        // Population publication runs after the fixed-step flight/contact kernels
        // but before the legacy combined render buffer has been assembled.
        cudaEvent_t populationInputsReady{},populationPackReady{};
        std::array<cudaEvent_t,SnowfallRingSize> populationCopiesDone{};
        std::array<bool,SnowfallRingSize> populationCopied{};
        std::array<uint32_t,3> groupCursors{};
        // Physical-only revision: descriptor changes, visibility, or any
        // potentially moving physical visual invalidate conservatively.
        uint64_t physicalDescriptorSignature{1469598103934665603ull};
        uint64_t publishedPhysicalDescriptorSignature{},physicalRayRevision{};
        bool physicalRayPublished{},physicalRayVisible{},physicalRayCanMove{};
        PublicationSlot::ResourceGroup& Group(uint32_t kind,uint32_t index){
            auto& s=slots[index];return kind==0?s.physicsVisual:(kind==1?s.population:s.snow);
        }
        // A frame may reserve the destination and fence before simulation.
        // The physics worker publishes into it after the current step finishes.
        PublicationSlot* pendingPublication{};
        uint32_t pendingPublicationIndex{};
        uint64_t pendingConsumerValue{};
        std::array<SharedBuffer,SnowfallRingSize> snowfallStates{};
        GpuParticleWindSample* snowfallWindCells{};
        uint32_t snowfallCurrentSlot{};
        bool snowfallReady{};
        double snowfallPendingSeconds{},snowfallTimeSeconds{};
        struct ProbeSlot {LabPhysicsStateV1* host{};uint32_t* overflow{};cudaEvent_t done{},gpuBegin{},gpuEnd{};bool pending{};uint64_t tick{},frame{};double seconds{};uint32_t count{};};
        std::array<ProbeSlot,3> probeSlots;uint32_t probeCursor{};uint64_t lastProbeFrame{},publicationSequence{};
        struct Timing {cudaEvent_t begin{},end{};bool pending{};uint64_t frame{},tick{},value{};};
        std::array<Timing,3> timings;uint32_t timingCursor{};
        std::array<Timing,48> stepTimings;uint32_t stepTimingCursor{};
        std::array<Timing,6> publicationTimings;uint32_t publicationTimingCursor{};
        std::array<Timing,24> particleTimings;uint32_t particleTimingCursor{};
        uint64_t particleSamples{};double particleSumMs{},particlePeakMs{};
        uint64_t publicationBackpressureCount{};
        double frameCommandWaitMs{},frameFetchBlockingMs{},frameTopologyWaitMs{};
        uint32_t frameUntimedSteps{};
        // Compact diagnostic, counters only; no GPU queries, waits or per-frame files.
        PhysicsDiagnosticProfile diagnostic;
        LabGpuPublicationProviderV1 provider{};std::vector<LabU32> streamMaterials;
        Scene(Impl& r,Owner& o,LabPhysicsSceneV1 h,const LabPhysicsSceneDescV1& d):runtime(r),owner(o),handle(h),desc(d){}
        void Initialize();void Destroy();
        Body* FindBody(LabPhysicsBodyV1 h){uint32_t i=uint32_t(h)-1;return h&&i<bodies.size()&&bodies[i].handle==h&&bodies[i].actor?&bodies[i]:nullptr;}
        Joint* FindJoint(LabPhysicsJointV1 h){uint32_t i=uint32_t(h)-1;return h&&i<joints.size()&&joints[i].handle==h&&joints[i].joint?&joints[i]:nullptr;}
        uint32_t CreateBody(const LabPhysicsBodyDescV1&,LabPhysicsBodyV1*);uint32_t DestroyBody(LabPhysicsBodyV1);
        uint32_t SetBodyCollisionEnabled(LabPhysicsBodyV1,bool);
        uint32_t SetBodySpeculativeCCD(LabPhysicsBodyV1,bool);
        uint32_t CreateD6(const LabPhysicsD6DescV1&,LabPhysicsJointV1*);uint32_t ConfigureD6(LabPhysicsJointV1,const LabPhysicsD6DescV1&);uint32_t DestroyJoint(LabPhysicsJointV1);
        uint32_t Pause(bool);uint32_t EditSnapshot(LabPhysicsEditSnapshotV1*);uint32_t Edit(const LabPhysicsEditBatchV1&);uint32_t Translate(const LabPhysicsBodyV1*,uint32_t,LabVec3V1);uint32_t ValidateEdit(LabPhysicsEditValidationV1*);void CaptureEditPoses();void WriteEditPoses();void CommitEditPoses();
        void RefreshIndices();void RefreshDescriptors();void ReadState(bool first);void Step(float,uint32_t);void Advance(double,uint64_t);
        Timing* BeginTiming(std::span<Timing>,uint32_t&,uint64_t=0,uint64_t=0,cudaStream_t=nullptr);
        void CollectTimings();
        uint32_t SetVisuals(const LabPhysicsVisualV1*,uint32_t,const LabPhysicsLightV1*,uint32_t);
        void DestroyPublications();
        uint32_t ReservePublication(LabGpuPublicationV1*,bool deferred);
        void CompletePublication();
        uint32_t Acquire(LabGpuPublicationV1*);void Release(void*,uint64_t);
        void QueueProbe();uint32_t Poll(LabPhysicsSnapshotV1*);
    };
    struct Owner {Impl* runtime{};uint64_t id{};LabGpuInteropServiceV1 interop{};LabPhysicsServiceV1 service{};std::map<LabPhysicsSceneV1,std::unique_ptr<Scene>> scenes;};
    std::map<uint64_t,std::unique_ptr<Owner>> owners;
    uint32_t Generation(){if(nextGeneration==UINT32_MAX)Fail("PhysX handle generations exhausted");return nextGeneration++;}
    uint64_t LeaseGeneration(){if(nextLeaseGeneration==UINT64_MAX)Fail("PhysX publication lease generations exhausted");return nextLeaseGeneration++;}
    Scene* Find(Owner& o,LabPhysicsSceneV1 h){auto it=o.scenes.find(h);return it==o.scenes.end()?nullptr:it->second.get();}
    void Check(){if(errors.failed.load()||!cuda||!cuda->contextIsValid())Fail("PhysX GPU contract failed; CPU fallback is forbidden (inspect physics.error)");}
};

} // namespace lab
