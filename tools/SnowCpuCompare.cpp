// Portable, deliberately isolated reference for two snowfall optimizations.
// No CUDA, DirectX, PhysX, OpenXR, or third-party dependencies.
// This is NOT a GPU performance benchmark and does not execute snow.hlsl.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kParticleCount = 2'000'000;
constexpr int kCellSide = 64;
constexpr int kPhaseBins = 32;
constexpr float kTau = 6.2831853f;
constexpr float kEyeOffset = .032f;

struct Particle {
    float x, y, z, phase;
    float vx, vy, vz, wind;
};
static_assert(sizeof(Particle) == 32);
struct WindSample { float flutter, swirl; };
struct Settings {
    int width = 200, height = 200, frames = 48, count = kParticleCount;
    std::filesystem::path output = "snow_cpu_output";
};

uint32_t Mix(uint32_t h) {
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    return h ^ (h >> 16);
}
float Fraction(uint32_t h) {
    return float(Mix(h) & 0x00ffffffu) * (1.f / 16777216.f);
}
float Clamp(float x, float low, float high) { return std::max(low, std::min(high, x)); }
int CellIndex(float x) {
    const float wrapped = x - std::floor((x + 128.f) / 256.f) * 256.f;
    return std::clamp(int((wrapped + 128.f) * (float(kCellSide) / 256.f)), 0, kCellSide - 1);
}

std::vector<Particle> InitialParticles(int count) {
    std::vector<Particle> result(static_cast<size_t>(count));
    const uint32_t seed = 0x76a019abu;
    for (int i = 0; i < count; ++i) {
        const uint32_t key = Mix(uint32_t(i) ^ seed);
        auto& p = result[size_t(i)];
        p.x = (Fraction(key + 11u) - .5f) * 256.f;
        p.y = .04f + Fraction(key + 29u) * 17.96f;
        p.z = (Fraction(key + 47u) - .5f) * 256.f;
        p.phase = Fraction(key + 67u);
        p.vx = (Fraction(key + 73u) - .5f) * .35f;
        p.vy = -(.65f + 1.15f * Fraction(key + 79u));
        p.vz = (Fraction(key + 97u) - .5f) * .35f;
        p.wind = .5f + Fraction(key + 103u);
    }
    return result;
}

// The reference reproduces the expressions in GpuParticleField.cu, with
// standard CPU sin/cos instead of CUDA's approximate fast intrinsics.
void Advance(std::vector<Particle>& particles, float dt, float time,
             const std::vector<WindSample>* cells) {
    for (auto& p : particles) {
        float flutter, swirl;
        if (cells) {
            const int x = CellIndex(p.x), z = CellIndex(p.z);
            const int phase = std::min(int(p.phase * float(kPhaseBins)), kPhaseBins - 1);
            const auto& wind = (*cells)[(size_t(z) * kCellSide + size_t(x)) * kPhaseBins + size_t(phase)];
            flutter = wind.flutter;
            swirl = wind.swirl;
        } else {
            const float phase = p.phase * kTau;
            flutter = std::sin(phase + time * .85f + p.z * .035f);
            swirl = std::cos(phase * .73f + time * .58f + p.x * .025f);
        }
        p.vx += (.14f + flutter * .22f * p.wind - p.vx * .20f) * dt;
        p.vz += (.07f + swirl * .19f * p.wind - p.vz * .20f) * dt;
        const float terminal = .65f + 1.15f * p.phase;
        p.vy += (-terminal - p.vy) * std::min(1.f, dt * 2.f);
        p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
        if (p.y < .04f) { p.y += 18.f; p.x += (p.phase - .5f) * 3.6f; p.z += (p.wind - .75f) * 3.2f; }
        if (p.x >= 128.f) p.x -= 256.f;
        if (p.x < -128.f) p.x += 256.f;
        if (p.z >= 128.f) p.z -= 256.f;
        if (p.z < -128.f) p.z += 256.f;
    }
}

void BuildCellWind(std::vector<WindSample>& cells, float time) {
    constexpr float width = 256.f / float(kCellSide);
    for (int z = 0; z < kCellSide; ++z) {
        const float centerZ = -128.f + (float(z) + .5f) * width;
        for (int x = 0; x < kCellSide; ++x) {
            const float centerX = -128.f + (float(x) + .5f) * width;
            for (int phase = 0; phase < kPhaseBins; ++phase) {
                const float p = (float(phase) + .5f) / float(kPhaseBins) * kTau;
                auto& sample = cells[(size_t(z) * kCellSide + size_t(x)) * kPhaseBins + size_t(phase)];
                sample.flutter = std::sin(p + time * .85f + centerZ * .035f);
                sample.swirl = std::cos(p * .73f + time * .58f + centerX * .025f);
            }
        }
    }
}

struct Image {
    int width, height;
    std::array<std::vector<uint32_t>, 2> coverage;
    explicit Image(int w, int h) : width(w), height(h) {
        for (auto& c : coverage) c.resize(size_t(w) * size_t(h));
    }
};

// A small orthographic-independent perspective projection of the snowfall
// coverage pass. One integer accumulation is used per pixel, per eye.
void Splat(const Particle& s, int eye, Image& image) {
    const float eyeX = eye ? kEyeOffset : -kEyeOffset;
    const float px = s.x + std::floor((eyeX - s.x + 128.f) / 256.f) * 256.f;
    const float pz = s.z + std::floor((-s.z + 128.f) / 256.f) * 256.f;
    if (std::abs(px) < 4.2f && std::abs(pz) < 4.2f && s.y < 4.05f) return;
    const float dx = px - eyeX, dy = s.y - 1.7f, dz = pz;
    const float distanceSq = dx * dx + dy * dy + dz * dz;
    if (std::abs(dx) > 35.f || std::abs(dz) > 35.f || distanceSq > 1225.f || dz <= .05f) return;
    const float projectedX = dx / dz;
    const float projectedY = dy / dz;
    if (std::abs(projectedX) >= 1.f || std::abs(projectedY) >= 1.f) return;
    const int x = int((projectedX * .5f + .5f) * image.width);
    const int y = int((.5f - projectedY * .5f) * image.height);
    if (x < 0 || y < 0 || x >= image.width || y >= image.height) return;
    // Proxy of the depth test: a simple opaque wall beyond the first 12 m.
    if (dz > 12.f && x > image.width * 2 / 3 && y > image.height / 3) return;
    const float distance = std::sqrt(distanceSq);
    const uint32_t strength = std::max(1u, uint32_t(330.f * Clamp((35.f - distance) / 15.f, 0.f, 1.f) * (.4f + .6f * s.phase)));
    auto& coverage = image.coverage[size_t(eye)];
    const size_t pixel = size_t(y) * size_t(image.width) + size_t(x);
    coverage[pixel] += strength;
    if (distance < 9.f && x + 1 < image.width) coverage[pixel + 1] += strength / 3u;
}

void Render(const std::vector<Particle>& particles, Image& image, bool stereoShared) {
    if (!stereoShared) {
        for (int eye = 0; eye != 2; ++eye)
            for (const auto& p : particles) Splat(p, eye, image);
    } else {
        for (const auto& p : particles) {
            // The roof test and other globally valid tests can be shared;
            // each eye still performs its own depth and projection tests.
            Splat(p, 0, image);
            Splat(p, 1, image);
        }
    }
}

std::vector<uint8_t> Colour(const Image& image, int eye) {
    std::vector<uint8_t> rgb(size_t(image.width) * size_t(image.height) * 3);
    for (int y = 0; y < image.height; ++y) for (int x = 0; x < image.width; ++x) {
        const size_t pixel = size_t(y) * size_t(image.width) + size_t(x);
        const size_t dst = pixel * 3;
        const float t = float(y) / float(image.height);
        const bool wall = x > image.width * 2 / 3 && y > image.height / 3;
        const float snow = Clamp(float(image.coverage[size_t(eye)][pixel]) / 850.f, 0.f, 1.f);
        const float nightR = wall ? 31.f : 12.f + 30.f * t;
        const float nightG = wall ? 27.f : 25.f + 28.f * t;
        const float nightB = wall ? 37.f : 53.f + 35.f * t;
        rgb[dst] = uint8_t(nightR + (192.f - nightR) * snow);
        rgb[dst + 1] = uint8_t(nightG + (212.f - nightG) * snow);
        rgb[dst + 2] = uint8_t(nightB + (240.f - nightB) * snow);
    }
    return rgb;
}

void WritePpm(const std::filesystem::path& file, const std::vector<uint8_t>& pixels, int w, int h) {
    std::ofstream out(file, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write " + file.string());
    out << "P6\n" << w << " " << h << "\n255\n";
    out.write(reinterpret_cast<const char*>(pixels.data()), std::streamsize(pixels.size()));
}

uint64_t Fingerprint(const std::vector<uint8_t>& pixels) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t byte : pixels) { h ^= byte; h *= 1099511628211ull; }
    return h;
}

struct Result {
    const char* name;
    double physicsMs{}, renderMs{};
    std::vector<uint64_t> hashes;
    std::vector<Particle> finalState;
    std::vector<std::vector<uint8_t>> pixels;
};

Result Run(const Settings& config, const char* name, bool stereoShared, bool cellWind) {
    auto particles = InitialParticles(config.count);
    std::vector<WindSample> cells;
    if (cellWind) cells.resize(size_t(kCellSide) * kCellSide * kPhaseBins);
    Result result; result.name = name;
    result.hashes.reserve(size_t(config.frames));
    result.pixels.reserve(size_t(config.frames));
    const auto output = config.output / name;
    std::filesystem::create_directories(output);
    using clock = std::chrono::steady_clock;
    for (int f = 0; f < config.frames; ++f) {
        const float dt = 2.f / float(config.frames);
        const float time = (float(f) + 1.f) * dt;
        const auto begin = clock::now();
        if (cellWind) BuildCellWind(cells, time);
        Advance(particles, dt, time, cellWind ? &cells : nullptr);
        const auto physicsEnd = clock::now();
        Image image(config.width, config.height);
        Render(particles, image, stereoShared);
        const auto renderEnd = clock::now();
        result.physicsMs += std::chrono::duration<double, std::milli>(physicsEnd - begin).count();
        result.renderMs += std::chrono::duration<double, std::milli>(renderEnd - physicsEnd).count();
        auto rgb = Colour(image, 0);
        result.hashes.push_back(Fingerprint(rgb));
        result.pixels.push_back(rgb);
        char frameName[32]; std::snprintf(frameName, sizeof(frameName), "frame_%03d.ppm", f);
        WritePpm(output / frameName, rgb, config.width, config.height);
    }
    result.finalState = std::move(particles);
    std::cout << name << ": simulation " << std::fixed << std::setprecision(1) << result.physicsMs
              << " ms; projection " << result.renderMs << " ms\n";
    return result;
}

void Report(const Settings& config, const std::vector<Result>& results) {
    if (results.size() != 4) throw std::logic_error("Expected four modes");
    const auto& baseline = results[0];
    const auto& stereo = results[1];
    const auto& cells = results[2];
    const auto& combined = results[3];
    const bool exactStereo = baseline.hashes == stereo.hashes &&
        std::memcmp(baseline.finalState.data(), stereo.finalState.data(),
                    baseline.finalState.size() * sizeof(Particle)) == 0;
    const bool exactCombinedVsCells = cells.hashes == combined.hashes &&
        std::memcmp(cells.finalState.data(), combined.finalState.data(),
                    cells.finalState.size() * sizeof(Particle)) == 0;
    if (!exactStereo || !exactCombinedVsCells)
        throw std::runtime_error("Stereo optimization changed simulation or rendered pixels");
    double imageAbs = 0.; uint64_t samples = 0;
    for (size_t frame = 0; frame < baseline.pixels.size(); ++frame)
        for (size_t i = 0; i < baseline.pixels[frame].size(); ++i) {
            imageAbs += std::abs(int(baseline.pixels[frame][i]) - int(combined.pixels[frame][i]));
            ++samples;
        }
    double errorSq = 0.;
    for (size_t i = 0; i < baseline.finalState.size(); ++i) {
        const auto& a = baseline.finalState[i], &b = combined.finalState[i];
        // Snow wraps every 256 m in X/Z; opposite sides of that periodic
        // boundary are physically adjacent, not 256 metres apart.
        double dx = double(a.x) - b.x, dy = double(a.y) - b.y, dz = double(a.z) - b.z;
        dx -= 256. * std::round(dx / 256.);
        dz -= 256. * std::round(dz / 256.);
        errorSq += dx * dx + dy * dy + dz * dz;
    }
    const double positionRms = std::sqrt(errorSq / double(config.count));
    const auto file = config.output / "report.json";
    std::ofstream out(file);
    if (!out) throw std::runtime_error("Cannot write report.json");
    out << std::fixed << std::setprecision(6);
    out << "{\n  \"reference_type\": \"CPU algorithmic proxy, NOT CUDA/DX12 benchmark\",\n";
    out << "  \"duration_seconds\": 2, \"frames\": " << config.frames
        << ", \"width\": " << config.width << ", \"height\": " << config.height
        << ", \"particles\": " << config.count << ",\n";
    out << "  \"cell_grid\": {\"side\": " << kCellSide << ", \"phase_bins\": " << kPhaseBins << "},\n";
    out << "  \"stereo_exact_images_and_state\": " << (exactStereo ? "true" : "false")
        << ", \"combined_matches_cell_only\": " << (exactCombinedVsCells ? "true" : "false") << ",\n";
    out << "  \"cell_approx_mean_pixel_abs_0_to_255\": " << imageAbs / double(samples)
        << ", \"cell_approx_final_position_rms_m\": " << positionRms << ",\n";
    out << "  \"modes\": {\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const double total = r.physicsMs + r.renderMs;
        out << "    \"" << r.name << "\": {\"physics_ms\": " << r.physicsMs
            << ", \"render_ms\": " << r.renderMs << ", \"total_ms\": " << total
            << ", \"relative_speedup\": " << (baseline.physicsMs + baseline.renderMs) / total
            << "}" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  }\n}\n";
    std::cout << "Stereo reference exact: " << (exactStereo ? "yes" : "NO")
              << "; cell field RMS position error: " << positionRms
              << " m; mean pixel difference: " << imageAbs / double(samples) << "/255\n";
}
}

int main(int argc, char** argv) {
    try {
        Settings cfg;
        for (int i = 1; i < argc; ++i) {
            if (i + 1 == argc) throw std::invalid_argument("Expected value after argument");
            const std::string opt = argv[i];
            const std::string val = argv[++i];
            if (opt == "--width") cfg.width = std::stoi(val);
            else if (opt == "--height") cfg.height = std::stoi(val);
            else if (opt == "--frames") cfg.frames = std::stoi(val);
            else if (opt == "--particles") cfg.count = std::stoi(val);
            else if (opt == "--out") cfg.output = val;
            else throw std::invalid_argument("Unknown option: " + opt);
        }
        if (cfg.width < 4 || cfg.height < 4 || cfg.frames < 1 || cfg.count < 1 || cfg.count > kParticleCount)
            throw std::invalid_argument("Invalid resolution, frame count or particle count");
        std::vector<Result> results;
        results.reserve(4);
        results.push_back(Run(cfg, "baseline", false, false));
        results.push_back(Run(cfg, "stereo", true, false));
        results.push_back(Run(cfg, "cells", false, true));
        results.push_back(Run(cfg, "combined", true, true));
        Report(cfg, results);
    } catch (const std::exception& e) {
        std::cerr << "SnowCpuCompare failed: " << e.what() << '\n';
        return 1;
    }
}
