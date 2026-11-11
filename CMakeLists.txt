cmake_minimum_required(VERSION 3.30)
cmake_policy(SET CMP0091 NEW)
# Set before compiler detection: command-line/environment defaults cannot select Ada.
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "LaboVR reference GPU: Blackwell SM120" FORCE)
project(LaboVR VERSION 1.1.9 LANGUAGES C CXX CUDA)
if(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Windows MSVC x64 required")
endif()
option(LABOVR_DIAGNOSTIC_TRACE "Compile detailed LaboVR diagnostic trace" ON)
set(LAB_MSAA_SAMPLES 1 CACHE STRING "")
if(NOT LAB_MSAA_SAMPLES MATCHES "^(1|4)$")
    message(FATAL_ERROR "LAB_MSAA_SAMPLES must be 1 or 4")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR "SM120 requires CUDA 12.8 or newer")
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
set(PHYSX_PRESET "vc17win64" CACHE STRING "" FORCE)
find_package(CUDAToolkit 12.8 REQUIRED)
file(GLOB LAB_CUDART_DLLS "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
list(SORT LAB_CUDART_DLLS COMPARE NATURAL ORDER DESCENDING)
list(LENGTH LAB_CUDART_DLLS LAB_CUDART_DLL_COUNT)
if(LAB_CUDART_DLL_COUNT EQUAL 0)
    message(FATAL_ERROR "CUDA runtime DLL missing from ${CUDAToolkit_BIN_DIR}")
endif()
list(GET LAB_CUDART_DLLS 0 LAB_CUDART_DLL)
include(FetchContent)
# Provision dependencies only when needed. The less strict
# FETCHCONTENT_UPDATES_DISCONNECTED does not disable URL downloads.
# On a previously completed configure, bypass *all* download/update/patch
# steps if every pinned source tree is present. This is particularly important
# for the large PhysX archive and the Windows Release iteration cycle.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update fetched Git dependencies" FORCE)
option(LABOVR_REFRESH_DEPENDENCIES "Explicitly re-fetch third-party dependencies (turn OFF again afterwards)" OFF)
set(_lab_vendor_packages openxr agility dxc steamaudio pix cgltf physx_lib ktx)
set(_lab_vendor_complete TRUE)
foreach(_lab_package IN LISTS _lab_vendor_packages)
    set(_lab_package_dir "${FETCHCONTENT_BASE_DIR}/${_lab_package}-src")
    if(NOT IS_DIRECTORY "${_lab_package_dir}")
        set(_lab_vendor_complete FALSE)
        break()
    endif()
    file(GLOB _lab_package_contents "${_lab_package_dir}/*")
    if(NOT _lab_package_contents)
        set(_lab_vendor_complete FALSE)
        break()
    endif()
endforeach()
if(NOT LABOVR_REFRESH_DEPENDENCIES AND _lab_vendor_complete
   AND EXISTS "${CMAKE_BINARY_DIR}/CMakeFiles/generate.stamp")
    set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "Reuse already configured local dependency sources" FORCE)
    message(STATUS "LaboVR dependencies: reusing cached sources, download disabled")
else()
    set(FETCHCONTENT_FULLY_DISCONNECTED OFF CACHE BOOL "Allow missing/explicitly refreshed third-party sources" FORCE)
    message(STATUS "LaboVR dependencies: initial provisioning or explicit refresh (network permitted)")
endif()
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_API_LAYERS OFF CACHE BOOL "" FORCE)
set(BUILD_CONFORMANCE_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SDK_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SPECIFICATION OFF CACHE BOOL "" FORCE)
set(BUILD_LOADER ON CACHE BOOL "" FORCE)
set(DYNAMIC_LOADER OFF CACHE BOOL "" FORCE)
FetchContent_Declare(openxr URL https://github.com/KhronosGroup/OpenXR-SDK/archive/refs/tags/release-1.1.63.zip DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(openxr)
FetchContent_Declare(agility URL https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.619.6 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(dxc URL https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DXC/1.9.2607.13 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(steamaudio URL https://github.com/ValveSoftware/steam-audio/releases/download/v4.6.1/steamaudio_4.6.1.zip DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(pix URL https://www.nuget.org/api/v2/package/WinPixEventRuntime/1.0.240308001 DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(cgltf URL https://github.com/jkuhlmann/cgltf/archive/refs/tags/v1.15.zip DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(agility dxc steamaudio cgltf pix)
set(PX_BUILDSNIPPETS OFF CACHE BOOL "" FORCE)
set(PX_BUILDPUBLICSAMPLES OFF CACHE BOOL "" FORCE)
set(PX_GENERATE_STATIC_LIBRARIES ON CACHE BOOL "" FORCE)
set(PX_GENERATE_GPU_PROJECTS ON CACHE BOOL "" FORCE)
set(PHYSX_BUILD_GPU_PROJECTS ON CACHE BOOL "" FORCE)
set(NV_USE_STATIC_WINCRT OFF CACHE BOOL "" FORCE)
set(NV_USE_DEBUG_WINCRT OFF CACHE BOOL "" FORCE)
set(PX_OUTPUT_LIB_DIR "${CMAKE_BINARY_DIR}/physx/lib" CACHE PATH "" FORCE)
set(PX_OUTPUT_BIN_DIR "${CMAKE_BINARY_DIR}/physx/bin" CACHE PATH "" FORCE)
set(TARGET_BUILD_PLATFORM windows CACHE STRING "" FORCE)
set(PX_GENERATE_GPU_PROJECTS_ONLY OFF CACHE BOOL "" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_LIBRARY_ROOT}" CACHE PATH "" FORCE)
FetchContent_Declare(physx_lib URL https://github.com/NVIDIA-Omniverse/PhysX/archive/refs/tags/110.1-omni-and-physx-5.9.0.zip SOURCE_SUBDIR physx DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    PATCH_COMMAND ${CMAKE_COMMAND} "-DPHYSX_SOURCE=<SOURCE_DIR>" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PhysXSm120.cmake")
FetchContent_MakeAvailable(physx_lib)
if(MSVC)
    function(lab_physx_msvc_compat directory)
        get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(target IN LISTS targets)
            get_target_property(target_type ${target} TYPE)
            if(NOT target_type STREQUAL "INTERFACE_LIBRARY" AND NOT target_type STREQUAL "UTILITY")
                target_compile_options(${target} PRIVATE
                    "$<$<COMPILE_LANGUAGE:CXX>:/wd5054>"
                    "$<$<COMPILE_LANGUAGE:CXX>:/wd5055>")
            endif()
        endforeach()
        get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
        foreach(child IN LISTS children)
            lab_physx_msvc_compat("${child}")
        endforeach()
    endfunction()
    lab_physx_msvc_compat("${physx_lib_SOURCE_DIR}/physx")
endif()
if(NOT TARGET physx_lib AND NOT TARGET PhysX)
    set(PHYSX_ROOT_DIR "${physx_lib_SOURCE_DIR}/physx" CACHE PATH "" FORCE)
    add_subdirectory("${PHYSX_ROOT_DIR}/compiler/public" "${physx_lib_BINARY_DIR}/public")
endif()
# PhysX uses both explicit --generate-code flags and target CUDA properties.
# Pin every generated target as well as our own kernels; fail the delivered binary
# check if an upstream change bypasses either path.
function(lab_pin_cuda directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES 120)
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        lab_pin_cuda("${child}")
    endforeach()
endfunction()
lab_pin_cuda("${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "LaboVR reference GPU: Blackwell SM120" FORCE)
if(TARGET physx_lib)
    set(LAB_PHYSX physx_lib)
else()
    set(LAB_PHYSX PhysX PhysXCommon PhysXFoundation PhysXExtensions PhysXCooking PhysXPvdSDK)
endif()
set(KTX_FEATURE_TESTS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_TOOLS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_DOC OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(KTX_FEATURE_GL_UPLOAD OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_VK_UPLOAD OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_LOADTEST_APPS OFF CACHE STRING "" FORCE)
FetchContent_Declare(ktx GIT_REPOSITORY https://github.com/KhronosGroup/KTX-Software.git GIT_TAG v4.4.2 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(ktx)
file(GLOB_RECURSE AGILITY_HEADERS "${agility_SOURCE_DIR}/*d3d12.h")
file(GLOB_RECURSE AGILITY_FILES "${agility_SOURCE_DIR}/*.dll")
file(GLOB_RECURSE DXC_HEADERS "${dxc_SOURCE_DIR}/*dxcapi.h")
list(GET DXC_HEADERS 0 DXC_HEADER)
get_filename_component(DXC_INCLUDE "${DXC_HEADER}" DIRECTORY)
file(GLOB_RECURSE DXC_FILES "${dxc_SOURCE_DIR}/*.exe" "${dxc_SOURCE_DIR}/*.dll")
file(GLOB_RECURSE PIX_HEADERS "${pix_SOURCE_DIR}/*pix3.h")
file(GLOB_RECURSE PIX_FILES "${pix_SOURCE_DIR}/*.lib" "${pix_SOURCE_DIR}/*.dll")
file(GLOB_RECURSE PHONON_HEADERS "${steamaudio_SOURCE_DIR}/*phonon.h")
file(GLOB_RECURSE PHONON_FILES "${steamaudio_SOURCE_DIR}/*.lib" "${steamaudio_SOURCE_DIR}/*.dll")
list(GET AGILITY_HEADERS 0 AGILITY_HEADER)
list(GET PHONON_HEADERS 0 PHONON_HEADER)
list(GET PIX_HEADERS 0 PIX_HEADER)
get_filename_component(PIX_INCLUDE "${PIX_HEADER}" DIRECTORY)
get_filename_component(AGILITY_INCLUDE "${AGILITY_HEADER}" DIRECTORY)
get_filename_component(PHONON_INCLUDE "${PHONON_HEADER}" DIRECTORY)
function(pick_x64 output name)
    string(TOLOWER "${name}" wanted)
    foreach(candidate IN LISTS ARGN)
        get_filename_component(base "${candidate}" NAME)
        string(TOLOWER "${base}" base)
        string(TOLOWER "${candidate}" path)
        if(base STREQUAL wanted AND path MATCHES "(^|[/\\])(x64|win64|amd64|windows-x64)([/\\]|$)")
            set(${output} "${candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "x64 ${name} missing")
endfunction()
pick_x64(D3D12CORE_DLL D3D12Core.dll ${AGILITY_FILES})
pick_x64(D3D12LAYERS_DLL D3D12SDKLayers.dll ${AGILITY_FILES})
pick_x64(DXC_EXE dxc.exe ${DXC_FILES})
pick_x64(DXCOMPILER_DLL dxcompiler.dll ${DXC_FILES})
pick_x64(DXIL_DLL dxil.dll ${DXC_FILES})
pick_x64(PHONON_LIB phonon.lib ${PHONON_FILES})
pick_x64(PHONON_DLL phonon.dll ${PHONON_FILES})
pick_x64(PIX_LIB WinPixEventRuntime.lib ${PIX_FILES})
pick_x64(PIX_DLL WinPixEventRuntime.dll ${PIX_FILES})
add_library(phonon SHARED IMPORTED GLOBAL)
set_target_properties(phonon PROPERTIES IMPORTED_IMPLIB "${PHONON_LIB}" IMPORTED_LOCATION "${PHONON_DLL}" INTERFACE_INCLUDE_DIRECTORIES "${PHONON_INCLUDE}")
add_library(pixruntime SHARED IMPORTED GLOBAL)
set_target_properties(pixruntime PROPERTIES IMPORTED_IMPLIB "${PIX_LIB}" IMPORTED_LOCATION "${PIX_DLL}" INTERFACE_INCLUDE_DIRECTORIES "${PIX_INCLUDE}")
add_executable(LaboVRCook labs/labovr/tools/Cook.cpp src/core/Assets.cpp src/core/SceneArchive.cpp src/core/ConsoleSpec.cpp)
target_compile_features(LaboVRCook PRIVATE cxx_std_20)
target_compile_definitions(LaboVRCook PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN KHRONOS_STATIC)
target_include_directories(LaboVRCook PRIVATE src/core "${cgltf_SOURCE_DIR}")
target_compile_options(LaboVRCook PRIVATE /O2 /utf-8 /arch:AVX2)
target_link_libraries(LaboVRCook PRIVATE ktx ole32 user32 windowscodecs)
set(SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
function(shader source entry profile)
    set(shader_source "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${source}")
    set(output "${SHADER_DIR}/${entry}.cso")
    set(shader_deps shaders/common.hlsli shaders/visual_common.hlsli shaders/optical_common.hlsli)
    if(source STREQUAL "scene.hlsl")
        list(APPEND shader_deps shaders/reflection_hit.hlsli)
    endif()
    add_custom_command(OUTPUT "${output}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_DIR}"
        COMMAND "${DXC_EXE}" -E ${entry} -T ${profile} -HV 2021 -D LAB_MSAA_SAMPLES=${LAB_MSAA_SAMPLES} -O3 -Zpr -Qstrip_debug -Qstrip_reflect -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders" -Fo "${output}" "${shader_source}"
        DEPENDS "${shader_source}" ${shader_deps} VERBATIM)
    set(SHADER_OUTPUTS ${SHADER_OUTPUTS} "${output}" PARENT_SCOPE)
endfunction()

shader(compute.hlsl MergeReflection cs_6_6)
shader(fxaa.hlsl FxaaMain cs_6_6)
shader(scene.hlsl MeshMain ms_6_6)
shader(scene.hlsl ShadowMain ms_6_6)
shader(scene.hlsl PixelMain ps_6_6)
shader(scene.hlsl DepthPixel ps_6_6)
shader(present.hlsl Fullscreen vs_6_6)
shader(present.hlsl Tonemap ps_6_6)
shader(home_menu.hlsl HomeMenuVS vs_6_6)
shader(home_menu.hlsl HomeMenuPS ps_6_6)
foreach(entry OpticalClear OpticalSplat OpticalPrefix OpticalOrigin)
    shader(optical.hlsl ${entry} cs_6_6)
endforeach()
shader(optical.hlsl OpticalDebugVS vs_6_6)
shader(optical.hlsl OpticalDebugPS ps_6_6)
shader(snow.hlsl SnowClear cs_6_6)
shader(snow.hlsl SnowSplat cs_6_6)
shader(snow.hlsl SnowSplatStereo cs_6_6)
shader(snow.hlsl SnowCompositeVS vs_6_6)
shader(snow.hlsl SnowCompositePS ps_6_6)
shader(atmosphere.hlsl FogDensity cs_6_6)
shader(atmosphere.hlsl FogIlluminate cs_6_6)
shader(atmosphere.hlsl FogIntegrate cs_6_6)
shader(atmosphere.hlsl FogVS vs_6_6)
shader(atmosphere.hlsl FogComposite ps_6_6)
shader(atmosphere.hlsl FlyShadowVS vs_6_6)
shader(atmosphere.hlsl FlyShadowPS ps_6_6)
shader(atmosphere.hlsl FlyProxyVS vs_6_6)
shader(atmosphere.hlsl FlyProxyPS ps_6_6)
foreach(entry CullMain RayBuildMain DepthMain HiZMain ClusterMain DenoiseMain DeformMain DeformBoundsMain HotDataMain)
    shader(compute.hlsl ${entry} cs_6_6)
endforeach()
foreach(entry VisualHeightVS VisualHeightHS VisualHeightDS VisualHeightPS)
    if(entry MATCHES "VS$")
        set(profile vs_6_6)
    elseif(entry MATCHES "HS$")
        set(profile hs_6_6)
    elseif(entry MATCHES "DS$")
        set(profile ds_6_6)
    else()
        set(profile ps_6_6)
    endif()
    shader(visual_height_pbr.hlsl ${entry} ${profile})
endforeach()
shader(visual_gel.hlsl VisualGelMeshVS vs_6_6)
shader(visual_gel.hlsl VisualGelStreamVS vs_6_6)
shader(visual_gel.hlsl VisualGelPS ps_6_6)
shader(visual_stream.hlsl VisualStreamVS vs_6_6)
shader(visual_stream.hlsl VisualStreamPS ps_6_6)
shader(visual_stream.hlsl VisualPointVS vs_6_6)
shader(visual_stream.hlsl VisualPointPS ps_6_6)
shader(visual_cosmos.hlsl VisualCosmosVS vs_6_6)
shader(visual_cosmos.hlsl VisualCosmosPS ps_6_6)
shader(visual_liquid_orb.hlsl VisualOrbVS vs_6_6)
shader(visual_liquid_orb.hlsl VisualOrbPS ps_6_6)
shader(visual_bloom.hlsl VisualBloomVS vs_6_6)
shader(visual_bloom.hlsl VisualBloomExtractPS ps_6_6)
shader(visual_bloom.hlsl VisualBloomBlurPS ps_6_6)
shader(visual_bloom.hlsl VisualBloomCompositePS ps_6_6)
add_custom_target(LaboVRShaders DEPENDS ${SHADER_OUTPUTS})
# Pipeline cache identity is intentionally independent from release labels and world source.
# It changes only when host GPU pipeline inputs change.
set(LAB_ABI_INPUT "LaboVR-PipelineABI16-Visual1-Scene3-SM66-RS11-CUDA120-Agility6196-DXC1.9-Zpr-O3-MSAA${LAB_MSAA_SAMPLES}-NVCC${CMAKE_CUDA_COMPILER_VERSION}-${CMAKE_CXX_COMPILER_ID}${CMAKE_CXX_COMPILER_VERSION}")
# Pipeline cache invalidation must follow GPU pipeline inputs only.
# Hashing all *.cpp/*.h made an ordinary physics/logging edit reconfigure
# CMake (including PhysX FetchContent), regenerate BuildConfig.h and rebuild
# the renderer. Keep the PSO cache conservative for renderer/ABI changes,
# while allowing true incremental builds for world, physics and logging edits.
set(LAB_ABI_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/Graphics.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/Graphics.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GraphicsInternal.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GraphicsFrameGraph.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GraphicsFrameGraph.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GraphicsResources.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GraphicsSetup.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GraphicsDomains.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GraphicsCapture.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GraphicsRender.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GpuRayScene.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GpuRayScene.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GpuSceneCache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GpuSceneCache.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GpuMemory.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GpuMemory.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/ConsoleSpec.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/GpuPublication.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/TerminalDomain.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/AtmosphereDomain.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/OpticalDomain.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/VisualDomain.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/VisualDomain.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/ClientLayers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/ClientLayers.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/ContentVisuals.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVRAtmosphere.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVRCapture.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVRGpuInteropService.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVRGpuService.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVROptical.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVRRendering.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVRTerminal.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVRVisual.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cartridge_api/LaboVRVisualService.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PhysXSm120.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
file(GLOB_RECURSE LAB_ABI_SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.hlsl"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.hlsli")
list(APPEND LAB_ABI_SOURCES ${LAB_ABI_SHADER_SOURCES})
list(SORT LAB_ABI_SOURCES)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${LAB_ABI_SOURCES})
foreach(source IN LISTS LAB_ABI_SOURCES)
    file(SHA256 "${source}" source_hash)
    string(APPEND LAB_ABI_INPUT "${source_hash}")
endforeach()
string(SHA256 LAB_PIPELINE_ABI "${LAB_ABI_INPUT}")
configure_file(src/core/BuildConfig.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/BuildConfig.h" @ONLY)
configure_file(cmake/PipelineIdentity.txt.in "${CMAKE_CURRENT_BINARY_DIR}/generated/PipelineIdentity.txt" @ONLY)
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/BuildVersion.h" "${CMAKE_CURRENT_BINARY_DIR}/generated/build_identity.json"
    COMMAND ${CMAKE_COMMAND}
        "-DVERSION_FILE=${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt"
        "-DPIPELINE_FILE=${CMAKE_CURRENT_BINARY_DIR}/generated/PipelineIdentity.txt"
        "-DOUTPUT_HEADER=${CMAKE_CURRENT_BINARY_DIR}/generated/BuildVersion.h"
        "-DOUTPUT_JSON=${CMAKE_CURRENT_BINARY_DIR}/generated/build_identity.json"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuildIdentity.cmake"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt" "${CMAKE_CURRENT_BINARY_DIR}/generated/PipelineIdentity.txt" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuildIdentity.cmake"
    VERBATIM)
add_custom_target(LaboVRBuildIdentity DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/generated/BuildVersion.h" "${CMAKE_CURRENT_BINARY_DIR}/generated/build_identity.json")
# Core compiles without a lab include directory or lab source file.
add_library(Core STATIC
    src/core/VrCapture.cpp src/core/SessionLog.cpp src/core/Agility.cpp src/core/InputMapping.cpp src/core/InputSystem.cpp src/core/XR.cpp src/core/Graphics.cpp src/core/GraphicsFrameGraph.cpp src/core/GraphicsResources.cpp src/core/GraphicsSetup.cpp src/core/GraphicsDomains.cpp src/core/GraphicsCapture.cpp src/core/GraphicsRender.cpp
    src/core/Audio.cpp src/core/Assets.cpp src/core/Jobs.cpp src/core/ConsoleSpec.cpp
    src/core/GpuMemory.cpp src/core/GpuSceneCache.cpp src/core/GpuRayScene.cpp src/core/GpuValidation.cpp
    src/core/Reverb.cpp src/core/RealtimeMemory.cpp
    src/core/Animation.cpp src/core/SceneArchive.cpp src/core/PhysicsRuntime.cpp src/core/PhysicsRuntimeBodies.cpp src/core/PhysicsRuntimePopulation.cpp src/core/PhysicsRuntimeSnowfall.cpp src/core/PhysicsDiagnostics.cpp src/core/PhysicsGpuInterop.cpp src/core/PhysicsCuda.cu src/core/PopulationCuda.cu src/core/GpuParticleField.cu src/core/SnowDomain.cpp src/core/TerminalDomain.cpp src/core/AtmosphereDomain.cpp src/core/OpticalDomain.cpp src/core/ProfileStorage.cpp)
target_include_directories(Core BEFORE PUBLIC "${AGILITY_INCLUDE}")
target_include_directories(Core PRIVATE "${physx_lib_SOURCE_DIR}/physx/include")
target_include_directories(Core PUBLIC src/core "${CMAKE_CURRENT_BINARY_DIR}/generated" PRIVATE "${cgltf_SOURCE_DIR}")
target_compile_definitions(Core PRIVATE PX_PHYSX_STATIC_LIB)
target_compile_definitions(Core PUBLIC LABOVR_DIAGNOSTIC_TRACE=$<BOOL:${LABOVR_DIAGNOSTIC_TRACE}> WIN32_LEAN_AND_MEAN NOMINMAX UNICODE _UNICODE _WIN32_WINNT=0x0A00 KHRONOS_STATIC _HAS_EXCEPTIONS=0)
target_compile_options(Core PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/W4;/utf-8;/permissive-;/O2;/GL;/GR-;/EHs-c-;/arch:AVX2>" "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=/utf-8,/O2,/MD>")
target_link_libraries(Core PUBLIC ${LAB_PHYSX} CUDA::cudart CUDA::cuda_driver ktx phonon pixruntime d3d12 dxgi dxguid xaudio2 avrt ole32 shell32 user32 advapi32 Synchronization windowscodecs openxr_loader)
# Force the SDK exports out of the static archive just as in the former EXE.
target_link_options(Core INTERFACE /INCLUDE:D3D12SDKVersion /INCLUDE:D3D12SDKPath)
set_target_properties(Core PROPERTIES CUDA_ARCHITECTURES 120)

# Procedural urban maps are generated out-of-source, then staged with the
# application. The generator uses Python's standard library only.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(URBAN_TEXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/urban_textures")
add_custom_command(
    OUTPUT "${URBAN_TEXTURE_DIR}/manifest.json"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/tools/UrbanTextures.py" "${URBAN_TEXTURE_DIR}" 512
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/tools/UrbanTextures.py"
    VERBATIM)
add_custom_target(UrbanTextures DEPENDS "${URBAN_TEXTURE_DIR}/manifest.json")
set(URBAN_MESH_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/urban_meshes")
add_custom_command(
    OUTPUT "${URBAN_MESH_DIR}/urban.gltf" "${URBAN_MESH_DIR}/urban.bin"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/tools/UrbanMeshes.py" "${URBAN_MESH_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/tools/UrbanMeshes.py"
    VERBATIM)
add_custom_target(UrbanMeshes DEPENDS "${URBAN_MESH_DIR}/urban.gltf" "${URBAN_MESH_DIR}/urban.bin")
set(URBAN_FLY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/urban_flies")
add_custom_command(
    OUTPUT "${URBAN_FLY_DIR}/flies.gltf" "${URBAN_FLY_DIR}/flies.bin"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/tools/UrbanFlyMeshes.py" "${URBAN_FLY_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/tools/UrbanFlyMeshes.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/assets/home_physics.gltf"
            "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/assets/home_physics.bin"
    VERBATIM)
add_custom_target(UrbanFlyMeshes DEPENDS "${URBAN_FLY_DIR}/flies.gltf" "${URBAN_FLY_DIR}/flies.bin")

# External cartridges are built and packaged outside ConsoleVR.

add_executable(LaboVR WIN32 labs/labovr/main.cpp labs/labovr/PerformanceCapture.cpp labs/labovr/PerformanceCapture.h
    labs/labovr/ContentCatalog.cpp labs/labovr/ClientAssets.cpp labs/labovr/ClientLayers.cpp labs/labovr/ContentVisuals.cpp labs/labovr/HomeWorld.cpp labs/labovr/HomePanel.cpp
    labs/labovr/HomeRendering.cpp labs/labovr/HomePhysics.cpp labs/labovr/UrbanLayout.cpp labs/labovr/UrbanSurface.cpp labs/labovr/UrbanScene.cpp labs/labovr/UrbanPhysics.cpp labs/labovr/HomeEditor.cpp labs/labovr/HomeProfile.cpp labs/labovr/LabInputMapping.cpp labs/labovr/LabAudio.cpp labs/labovr/PluginHost.cpp labs/labovr/VisualDomain.cpp labs/labovr/VisualAssets.cpp labs/labovr/PluginLoader.cpp labs/labovr/PluginWorldAdapter.cpp)
target_compile_definitions(LaboVR PRIVATE LABOVR_DIAGNOSTIC_TRACE=$<BOOL:${LABOVR_DIAGNOSTIC_TRACE}>)
target_include_directories(LaboVR PRIVATE "${DXC_INCLUDE}" labs/labovr src/cartridge_api)
target_link_libraries(LaboVR PRIVATE Core)
if(NOT TARGET PhysXGpu)
    message(FATAL_ERROR "PhysXGpu SM120 target missing; CPU physics is not a supported substitute")
endif()
add_dependencies(LaboVR PhysXGpu LaboVRShaders LaboVRCook LaboVRBuildIdentity UrbanTextures UrbanMeshes UrbanFlyMeshes)
target_link_options(LaboVR PRIVATE /LTCG /OPT:REF /OPT:ICF)
set_target_properties(LaboVR PROPERTIES CUDA_ARCHITECTURES 120 RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:LaboVR>")
add_custom_command(TARGET LaboVR POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:LaboVR>/D3D12" "$<TARGET_FILE_DIR:LaboVR>/assets" "$<TARGET_FILE_DIR:LaboVR>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${D3D12CORE_DLL}" "$<TARGET_FILE_DIR:LaboVR>/D3D12/D3D12Core.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${D3D12LAYERS_DLL}" "$<TARGET_FILE_DIR:LaboVR>/D3D12/D3D12SDKLayers.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PIX_DLL}" "$<TARGET_FILE_DIR:LaboVR>/WinPixEventRuntime.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:LaboVRCook>" "$<TARGET_FILE_DIR:LaboVR>/LaboVRCook.exe"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/input.ini" "$<TARGET_FILE_DIR:LaboVR>/input.ini"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:PhysXGpu>" "$<TARGET_FILE_DIR:LaboVR>/PhysXGpu_64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LAB_CUDART_DLL}" "$<TARGET_FILE_DIR:LaboVR>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PHONON_DLL}" "$<TARGET_FILE_DIR:LaboVR>/phonon.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXCOMPILER_DLL}" "$<TARGET_FILE_DIR:LaboVR>/dxcompiler.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXIL_DLL}" "$<TARGET_FILE_DIR:LaboVR>/dxil.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:ktx>" "$<TARGET_FILE_DIR:LaboVR>/ktx.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/assets" "$<TARGET_FILE_DIR:LaboVR>/assets"
     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${URBAN_TEXTURE_DIR}" "$<TARGET_FILE_DIR:LaboVR>/assets/urban/textures"
     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${URBAN_MESH_DIR}" "$<TARGET_FILE_DIR:LaboVR>/assets/urban"
     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${URBAN_FLY_DIR}" "$<TARGET_FILE_DIR:LaboVR>/assets/urban/flies"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${SHADER_DIR}" "$<TARGET_FILE_DIR:LaboVR>/shaders"
    COMMAND ${CMAKE_COMMAND} "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/profiles" "-DDESTINATION=$<TARGET_FILE_DIR:LaboVR>/profiles" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageProfiles.cmake"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/generated/build_identity.json" "$<TARGET_FILE_DIR:LaboVR>/build_identity.json"
    VERBATIM)
# Fast runtime staging: resource/config/shader changes do not need an EXE relink.
add_custom_target(StageLaboVR
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:LaboVR>/D3D12" "$<TARGET_FILE_DIR:LaboVR>/assets" "$<TARGET_FILE_DIR:LaboVR>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${D3D12CORE_DLL}" "$<TARGET_FILE_DIR:LaboVR>/D3D12/D3D12Core.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${D3D12LAYERS_DLL}" "$<TARGET_FILE_DIR:LaboVR>/D3D12/D3D12SDKLayers.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PIX_DLL}" "$<TARGET_FILE_DIR:LaboVR>/WinPixEventRuntime.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:LaboVRCook>" "$<TARGET_FILE_DIR:LaboVR>/LaboVRCook.exe"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/input.ini" "$<TARGET_FILE_DIR:LaboVR>/input.ini"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:PhysXGpu>" "$<TARGET_FILE_DIR:LaboVR>/PhysXGpu_64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LAB_CUDART_DLL}" "$<TARGET_FILE_DIR:LaboVR>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PHONON_DLL}" "$<TARGET_FILE_DIR:LaboVR>/phonon.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXCOMPILER_DLL}" "$<TARGET_FILE_DIR:LaboVR>/dxcompiler.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DXIL_DLL}" "$<TARGET_FILE_DIR:LaboVR>/dxil.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:ktx>" "$<TARGET_FILE_DIR:LaboVR>/ktx.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${CMAKE_CURRENT_SOURCE_DIR}/labs/labovr/assets" "$<TARGET_FILE_DIR:LaboVR>/assets"
     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${URBAN_TEXTURE_DIR}" "$<TARGET_FILE_DIR:LaboVR>/assets/urban/textures"
     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${URBAN_MESH_DIR}" "$<TARGET_FILE_DIR:LaboVR>/assets/urban"
     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${URBAN_FLY_DIR}" "$<TARGET_FILE_DIR:LaboVR>/assets/urban/flies"
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${SHADER_DIR}" "$<TARGET_FILE_DIR:LaboVR>/shaders"
    COMMAND ${CMAKE_COMMAND} "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/profiles" "-DDESTINATION=$<TARGET_FILE_DIR:LaboVR>/profiles" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageProfiles.cmake"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_BINARY_DIR}/generated/build_identity.json" "$<TARGET_FILE_DIR:LaboVR>/build_identity.json"
    DEPENDS LaboVR
    VERBATIM)

add_custom_target(LaboVRPackage)
add_dependencies(LaboVRPackage LaboVR StageLaboVR)
add_custom_target(LaboVRRuntime DEPENDS LaboVRPackage)
