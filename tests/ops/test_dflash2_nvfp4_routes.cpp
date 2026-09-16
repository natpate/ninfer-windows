// Oracle tests for the NVFP4 DFlash2 draft routes: the three-output attention input projection,
// the dynamic grouped conv prepare/finish pair, and the NVFP4 context_kv_materialize kernels.
// Every oracle evaluates the complete logical formula in FP64 from the represented public
// inputs, decoding the packed weights through quantized_weight::logical_weight_fp64.
#include <cuda_runtime.h>

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ninfer/ops/dynamic_grouped_conv.h"

#include "core/cyclic_kv_cache.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using quantized_weight::PackedWeight;
using quantized_weight::PatternedWeightOptions;

namespace {

constexpr int kHidden = 5120;
constexpr double kRelative = 2.0e-2;

DeviceBuffer payload_device(const std::vector<std::uint8_t>& payload) {
    DeviceBuffer buffer(payload.size());
    buffer.copy_from_host(payload.data(), payload.size());
    return buffer;
}

float half_to_f32(std::uint16_t bits) {
    const std::uint32_t sign = (bits >> 15) & 1U;
    const std::uint32_t exponent = (bits >> 10) & 0x1fU;
    const std::uint32_t fraction = bits & 0x3ffU;
    float value = 0.0F;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(fraction), -24);
    } else if (exponent != 31) {
        value = std::ldexp(static_cast<float>(fraction | 0x400U), static_cast<int>(exponent) - 25);
    }
    return sign != 0U ? -value : value;
}

float pattern_value(std::uint32_t seed, std::int32_t row, std::int32_t column, float scale) {
    const std::uint32_t mixed = seed * 747796405U + static_cast<std::uint32_t>(row) * 2891336453U +
                                static_cast<std::uint32_t>(column) * 19349663U;
    const int centered = static_cast<int>((mixed >> 13U) % 509U) - 254;
    return bf16_to_f32(f32_to_bf16(static_cast<float>(centered) * scale));
}

bool sample_ok(double actual, double reference, double tolerance) {
    return std::fabs(actual - reference) <= std::max(tolerance, std::fabs(reference) * kRelative);
}

PackedWeight nvfp4_weight(std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    PatternedWeightOptions options;
    options.weight_scale_divisor = 512.0F;
    options.input_scale_divisor  = 1.0F;
    return quantized_weight::make_patterned_weight(QType::NVFP4, rows, columns, seed, options);
}

// ---------------------------------------------------------------------------
// Three-output attention input projection.
// ---------------------------------------------------------------------------
int verify_attn_input() {
    const PackedWeight packed = nvfp4_weight(6144, kHidden, 501U);
    DeviceBuffer weight_device = payload_device(packed.payload);
    const Weight weight = packed.device_weight(weight_device.p);
    int failures        = 0;
    for (const std::int32_t tokens : {1, 8, 33}) {
        std::vector<float> host(static_cast<std::size_t>(kHidden) * tokens);
        for (std::int32_t column = 0; column < tokens; ++column) {
            for (std::int32_t row = 0; row < kHidden; ++row) {
                host[static_cast<std::size_t>(column) * kHidden + row] =
                    pattern_value(97U, row, column, 1.0F / 256.0F);
            }
        }
        DeviceBuffer x_device = to_device_bf16(host);
        DeviceBuffer q_device(static_cast<std::size_t>(4096) * tokens * sizeof(std::uint16_t));
        DeviceBuffer k_device(static_cast<std::size_t>(1024) * tokens * sizeof(std::uint16_t));
        DeviceBuffer v_device(static_cast<std::size_t>(1024) * tokens * sizeof(std::uint16_t));
        Tensor x(x_device.p, DType::BF16, {kHidden, tokens});
        Tensor q(q_device.p, DType::BF16, {4096, tokens});
        Tensor k(k_device.p, DType::BF16, {1024, tokens});
        Tensor v(v_device.p, DType::BF16, {1024, tokens});
        ops::attn_input_proj(x, weight, q, k, v, nullptr);
        cuda_synchronize();
        const auto q_got = from_device_bf16(q_device, static_cast<std::size_t>(4096) * tokens);
        const auto k_got = from_device_bf16(k_device, static_cast<std::size_t>(1024) * tokens);
        const auto v_got = from_device_bf16(v_device, static_cast<std::size_t>(1024) * tokens);
        const std::int32_t samples[] = {0, 123, 4095, 4096, 4600, 5120, 6143};
        for (const std::int32_t column : {0, tokens - 1}) {
            for (const std::int32_t row : samples) {
                double reference = 0.0;
                for (std::int32_t input = 0; input < kHidden; ++input) {
                    reference += quantized_weight::logical_weight_fp64(packed, row, input) *
                                 host[static_cast<std::size_t>(column) * kHidden + input];
                }
                const float actual = row < 4096   ? q_got[column * 4096 + row]
                                    : row < 5120 ? k_got[column * 1024 + (row - 4096)]
                                                 : v_got[column * 1024 + (row - 5120)];
                if (!sample_ok(actual, reference, 2.0e-2)) {
                    std::cerr << "attn_input T=" << tokens << " row=" << row
                              << " column=" << column << ": actual=" << actual << " reference="
                              << reference << '\n';
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------
// Dynamic grouped conv prepare + finish.
// ---------------------------------------------------------------------------
int verify_dynamic_conv(const PackedWeight& kernel_projection, const PackedWeight& projection,
                        std::int32_t width, std::int32_t batch, std::uint32_t seed) {
    constexpr int kCoefficientRows = 1280;
    const std::size_t columns      = static_cast<std::size_t>(width) * batch;
    std::vector<float> residual(columns * kHidden);
    std::vector<float> norm(kHidden);
    std::vector<float> base(kHidden * 4);
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::int32_t row = 0; row < kHidden; ++row) {
            residual[column * kHidden + row] =
                pattern_value(seed, row, static_cast<std::int32_t>(column), 1.0F / 128.0F);
        }
    }
    for (std::int32_t row = 0; row < kHidden; ++row) {
        norm[row] = bf16_to_f32(f32_to_bf16(0.75F + static_cast<float>(row % 17) * (1.0F / 128.0F)));
        for (int block = 0; block < 4; ++block) {
            base[static_cast<std::size_t>(block) * kHidden + row] =
                bf16_to_f32(f32_to_bf16(static_cast<float>((row * 3 + block) % 11) * 0.03125F));
        }
    }

    DeviceBuffer weight_device      = payload_device(kernel_projection.payload);
    DeviceBuffer projection_device  = payload_device(projection.payload);
    DeviceBuffer residual_device = to_device_bf16(residual);
    DeviceBuffer norm_device     = to_device_bf16(norm);
    DeviceBuffer base_device     = to_device_bf16(base);
    GuardedDeviceBuffer prepared_device(columns * kHidden * sizeof(std::uint16_t));
    GuardedDeviceBuffer finish_device(columns * 2 * 320 * sizeof(std::uint16_t));
    const std::size_t prepare_bytes = ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
        width, width, batch, batch);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(prepare_bytes, 1));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    Tensor residual_tensor(residual_device.p, DType::BF16, {kHidden, width, batch});
    Tensor norm_tensor(norm_device.p, DType::BF16, {kHidden});
    Tensor base_tensor(base_device.p, DType::BF16, {kHidden, 2, 2});
    Tensor prepared_tensor(prepared_device.data(), DType::BF16, {kHidden, width, batch});
    Tensor finish_tensor(finish_device.data(), DType::BF16, {320, 2, width, batch});
    ops::rmsnorm_dynamic_grouped_conv_prepare(residual_tensor, norm_tensor, 1.0e-6F, base_tensor,
                                              kernel_projection.device_weight(weight_device.p),
                                              prepared_tensor, finish_tensor, workspace, nullptr);
    cuda_synchronize();

    // FP64 oracle: rmsnorm, projected coefficients, in-place conv application, finish delta.
    std::vector<double> normed(columns * kHidden);
    for (std::size_t column = 0; column < columns; ++column) {
        double sum = 0.0;
        for (std::int32_t row = 0; row < kHidden; ++row) {
            const double value = residual[column * kHidden + row];
            sum += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum / kHidden + 1.0e-6);
        for (std::int32_t row = 0; row < kHidden; ++row) {
            normed[column * kHidden + row] =
                residual[column * kHidden + row] * inverse * norm[row];
        }
    }
    std::vector<double> projected(columns * kCoefficientRows);
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::int32_t row = 0; row < kCoefficientRows; ++row) {
            double sum = 0.0;
            for (std::int32_t input = 0; input < kHidden; ++input) {
                sum += quantized_weight::logical_weight_fp64(kernel_projection, row, input) *
                       normed[column * kHidden + input];
            }
            projected[column * kCoefficientRows + row] = sum;
        }
    }
    int failures = 0;
    const auto prepared_got = from_device_bf16(prepared_device.data(), columns * kHidden);
    const auto finish_got   = from_device_bf16(finish_device.data(), columns * 2 * 320);
    for (std::size_t column = 0; column < columns; ++column) {
        const std::int32_t position = static_cast<std::int32_t>(column % width);
        for (std::int32_t row = 0; row < kHidden; row += 337) {
            const std::int32_t group = row / 16;
            double value = (base[row] + projected[column * kCoefficientRows + group]) *
                           normed[column * kHidden + row];
            if (position > 0) {
                value += (base[kHidden + row] +
                          projected[column * kCoefficientRows + 320 + group]) *
                         normed[(column - 1) * kHidden + row];
            }
            if (!sample_ok(prepared_got[column * kHidden + row], value, 4.0e-2)) {
                std::cerr << "conv prepare W=" << width << " B=" << batch << " column=" << column
                          << " row=" << row << ": actual=" << prepared_got[column * kHidden + row]
                          << " reference=" << value << '\n';
                ++failures;
            }
        }
        for (const std::int32_t group : {0, 137, 319}) {
            for (int tap = 0; tap < 2; ++tap) {
                const double reference =
                    projected[column * kCoefficientRows + (2 + tap) * 320 + group];
                const float actual = finish_got[(column * 2 + tap) * 320 + group];
                if (!sample_ok(actual, reference, 3.0e-3)) {
                    std::cerr << "conv finish delta W=" << width << " column=" << column
                              << " tap=" << tap << " group=" << group
                              << ": actual=" << actual << " reference=" << reference << '\n';
                    ++failures;
                }
            }
        }
    }
    if (failures != 0) { return failures; }

    // Finish: linear_dynamic_grouped_conv_add with the NVFP4 projection parent.
    const std::int32_t input_rows = static_cast<std::int32_t>(projection.weight.k);
    std::vector<float> input(columns * input_rows);
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::int32_t row = 0; row < input_rows; ++row) {
            input[column * input_rows + row] =
                pattern_value(seed + 31U, row, static_cast<std::int32_t>(column), 1.0F / 128.0F);
        }
    }
    DeviceBuffer input_device = to_device_bf16(input);
    GuardedDeviceBuffer residual_out_device(columns * kHidden * sizeof(std::uint16_t));
    residual_out_device.fill(0);
    const std::size_t add_bytes = ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
        input_rows, width, width, batch, batch);
    GuardedDeviceBuffer add_scratch(std::max<std::size_t>(add_bytes, 1));
    WorkspaceArena add_workspace(DeviceSpan{add_scratch.data(), add_scratch.bytes()});
    Tensor input_tensor(input_device.p, DType::BF16, {input_rows, width, batch});
    Tensor residual_out_tensor(residual_out_device.data(), DType::BF16, {kHidden, width, batch});
    ops::linear_dynamic_grouped_conv_add(input_tensor,
                                         projection.device_weight(projection_device.p),
                                         base_tensor, finish_tensor, residual_out_tensor,
                                         add_workspace, nullptr);
    cuda_synchronize();
    const auto out_got = from_device_bf16(residual_out_device.data(), columns * kHidden);
    for (std::size_t column = 0; column < columns; ++column) {
        const std::int32_t position = static_cast<std::int32_t>(column % width);
        for (std::int32_t row = 0; row < kHidden; row += 337) {
            const std::int32_t group = row / 16;
            double z = 0.0;
            for (std::int32_t input_row = 0; input_row < input_rows; input_row += 3) {
                z += quantized_weight::logical_weight_fp64(projection, row, input_row) *
                     input[column * input_rows + input_row] * 3.0;
            }
            // The sampled-dot reconstruction above is exact only for all inputs; use the full dot.
            z = 0.0;
            for (std::int32_t input_row = 0; input_row < input_rows; ++input_row) {
                z += quantized_weight::logical_weight_fp64(projection, row, input_row) *
                     input[column * input_rows + input_row];
            }
            double value = (base[2 * kHidden + row] +
                            finish_got[(column * 2 + 0) * 320 + group]) *
                           z;
            if (position > 0) {
                double z_previous = 0.0;
                for (std::int32_t input_row = 0; input_row < input_rows; ++input_row) {
                    z_previous += quantized_weight::logical_weight_fp64(projection, row, input_row) *
                                  input[(column - 1) * input_rows + input_row];
                }
                value += (base[3 * kHidden + row] + finish_got[(column * 2 + 1) * 320 + group]) *
                         z_previous;
            }
            if (!sample_ok(out_got[column * kHidden + row], value, 8.0e-2)) {
                std::cerr << "conv add W=" << width << " B=" << batch << " column=" << column
                          << " row=" << row << ": actual=" << out_got[column * kHidden + row]
                          << " reference=" << value << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------
// NVFP4 context_kv_materialize.
// ---------------------------------------------------------------------------
constexpr int kCachePadded    = 2056;
constexpr int kCacheLanes     = 2;
constexpr double kRopeTheta   = 1.0e7;
constexpr std::uint16_t kCacheSentinel = 0x5a5aU;

std::size_t cache_elements() {
    return static_cast<std::size_t>(128) * kCachePadded * 8 * kCacheLanes;
}

double rope_angle(std::int32_t position, std::int32_t pair) {
    return static_cast<double>(position) * std::pow(kRopeTheta, -static_cast<double>(pair) / 64.0);
}

int verify_context_kv(std::int32_t width, std::int32_t batch, std::uint32_t seed, bool graph) {
    constexpr int kLayers = static_cast<int>(ops::kContextKVMaterializeLayers);
    const std::size_t columns = static_cast<std::size_t>(width) * batch;
    std::array<PackedWeight, kLayers> keys;
    std::array<PackedWeight, kLayers> values;
    std::array<DeviceBuffer, kLayers> key_device;
    std::array<DeviceBuffer, kLayers> value_device;
    std::vector<std::vector<float>> norms(kLayers);
    std::vector<DeviceBuffer> norm_device(kLayers);
    for (int layer = 0; layer < kLayers; ++layer) {
        keys[layer]   = nvfp4_weight(1024, kHidden, 601U + 2U * layer);
        values[layer] = nvfp4_weight(1024, kHidden, 602U + 2U * layer);
        key_device[layer]   = payload_device(keys[layer].payload);
        value_device[layer] = payload_device(values[layer].payload);
        norms[layer].resize(128);
        for (int dim = 0; dim < 128; ++dim) {
            norms[layer][dim] =
                bf16_to_f32(f32_to_bf16(0.75F + static_cast<float>((dim * 7 + layer) % 19) *
                                                         (1.0F / 128.0F)));
        }
        norm_device[layer] = to_device_bf16(norms[layer]);
    }
    std::vector<float> context(columns * kHidden);
    std::vector<std::int32_t> positions(columns);
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::int32_t row = 0; row < kHidden; ++row) {
            context[column * kHidden + row] =
                pattern_value(seed, row, static_cast<std::int32_t>(column), 1.0F / 128.0F);
        }
        positions[column] = 17 + static_cast<std::int32_t>(column) * 13;
    }
    const std::vector<std::int32_t> counts(batch, width);
    const std::vector<std::int32_t> slots = {0, 1};

    DeviceBuffer context_device = to_device_bf16(context);
    DeviceBuffer positions_device = to_device_i32(positions);
    DeviceBuffer counts_device    = to_device_i32(counts);
    DeviceBuffer slots_device     = to_device_i32(slots);
    std::vector<GuardedDeviceBuffer> cache_k;
    std::vector<GuardedDeviceBuffer> cache_v;
    std::array<ops::ContextKVMaterializeLayerView, kLayers> views;
    for (int layer = 0; layer < kLayers; ++layer) {
        cache_k.emplace_back(cache_elements() * sizeof(std::uint16_t));
        cache_v.emplace_back(cache_elements() * sizeof(std::uint16_t));
        cache_k[layer].fill(0x5a);
        cache_v[layer].fill(0x5a);
        views[layer] = ops::ContextKVMaterializeLayerView{
            keys[layer].device_weight(key_device[layer].p),
            values[layer].device_weight(value_device[layer].p),
            Tensor(norm_device[layer].p, DType::BF16, {128}),
            CyclicKVCacheLayerView{
                Tensor(cache_k[layer].data(), DType::BF16, {128, kCachePadded, 8, kCacheLanes}),
                Tensor(cache_v[layer].data(), DType::FP16, {128, kCachePadded, 8, kCacheLanes}),
                2048U, static_cast<std::uint32_t>(kCachePadded), 8U, 128U,
                static_cast<std::uint32_t>(kCacheLanes)},
        };
    }
    const std::size_t capacity = ops::context_kv_materialize_workspace_capacity_bytes(
        batch, width, width);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
    Tensor context_tensor(context_device.p, DType::BF16, {kHidden, width, batch});
    Tensor positions_tensor(positions_device.p, DType::I32, {width, batch});
    Tensor counts_tensor(counts_device.p, DType::I32, {batch});
    Tensor slots_tensor(slots_device.p, DType::I32, {batch});
    if (!graph) {
        ops::context_kv_materialize(context_tensor, positions_tensor, counts_tensor, slots_tensor,
                                    views, {0U, static_cast<std::uint32_t>(width)}, workspace,
                                    nullptr);
        cuda_synchronize();
    } else {
        // Mirror the W8 test's replay contract: capture with the b%3 mixed-count pattern in the
        // device buffers, then replay through the recorded executable with full counts, fresh
        // slots, and shifted positions.
        std::vector<std::int32_t> capture_counts(batch), capture_slots(batch, -1);
        std::vector<std::int32_t> capture_positions(columns, -1);
        for (std::int32_t b = 0; b < batch; ++b) {
            capture_counts[b] = b == batch - 1 || b % 3 == 2
                                    ? width
                                    : b % 3 == 1 ? std::max(1, width / 2) : 0;
            if (capture_counts[b] != 0) {
                capture_slots[b] = (b * 3 + 2) % 8;
                for (std::int32_t i = 0; i < capture_counts[b]; ++i) {
                    capture_positions[static_cast<std::size_t>(b) * width + i] = 2046 + b * 4096 + i;
                }
            }
        }
        counts_device.copy_from_host(capture_counts.data(), batch * 4);
        slots_device.copy_from_host(capture_slots.data(), batch * 4);
        positions_device.copy_from_host(capture_positions.data(), columns * 4);
        for (auto& buffer : cache_k) { buffer.fill(0x5a); }
        for (auto& buffer : cache_v) { buffer.fill(0x5a); }
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "context graph stream");
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        cudaStream_t capture_stream = stream;
        definition.capture(capture_stream, [&] {
            ops::context_kv_materialize(context_tensor, positions_tensor, counts_tensor,
                                        slots_tensor, views,
                                        {0U, static_cast<std::uint32_t>(width)}, workspace,
                                        capture_stream);
        });
        executable.instantiate(definition);
        counts_device.copy_from_host(counts.data(), batch * 4);
        slots_device.copy_from_host(slots.data(), batch * 4);
        positions_device.copy_from_host(positions.data(), columns * 4);
        for (auto& buffer : cache_k) { buffer.fill(0x5a); }
        for (auto& buffer : cache_v) { buffer.fill(0x5a); }
        cuda_synchronize();
        executable.launch(stream);
        cuda_synchronize(stream);
        cuda_check(cudaStreamDestroy(stream), "destroy context graph stream");
    }

    int failures = 0;

    for (int layer = 0; layer < kLayers; ++layer) {
        const auto k_got = from_device_bf16(cache_k[layer].data(), cache_elements());
        const auto v_raw = from_device<std::uint16_t>(cache_v[layer].data(), cache_elements());
        std::vector<float> v_got(cache_elements());
        for (std::size_t index = 0; index < cache_elements(); ++index) {
            v_got[index] = half_to_f32(v_raw[index]);
        }
        // Full FP64 oracle for sampled (column, head, dim) cells.
        for (std::size_t column = 0; column < columns; column += std::max<std::size_t>(1, columns / 3)) {
            const std::int32_t position = positions[column];
            const std::size_t slot      = static_cast<std::size_t>(slots[column / width]);
            for (const std::int32_t head : {0, 3, 7}) {
                std::vector<double> key_raw(128), key_normed(128);
                for (int dim = 0; dim < 128; ++dim) {
                    const std::int32_t row = head * 128 + dim;
                    double sum             = 0.0;
                    for (std::int32_t input = 0; input < kHidden; ++input) {
                        sum += quantized_weight::logical_weight_fp64(
                                   keys[layer], row, input) *
                               context[column * kHidden + input];
                    }
                    key_raw[dim] = sum;
                }
                double square = 0.0;
                for (int dim = 0; dim < 128; ++dim) { square += key_raw[dim] * key_raw[dim]; }
                const double inverse = 1.0 / std::sqrt(square / 128.0 + 1.0e-6);
                for (int dim = 0; dim < 128; ++dim) {
                    key_normed[dim] = key_raw[dim] * inverse * norms[layer][dim];
                }
                for (const std::int32_t pair : {0, 1, 31}) {
                    const double a_x = rope_angle(position, pair);
                    const double a_y = rope_angle(position, pair + 1);
                    const double kx =
                        key_normed[pair] * std::cos(a_x) - key_normed[pair + 64] * std::sin(a_x);
                    const double ky =
                        key_normed[pair + 64] * std::cos(a_x) + key_normed[pair] * std::sin(a_x);
                    const double kx1 = key_normed[pair + 1] * std::cos(a_y) -
                                       key_normed[pair + 65] * std::sin(a_y);
                    const double ky1 = key_normed[pair + 65] * std::cos(a_y) +
                                       key_normed[pair + 1] * std::sin(a_y);
                    const std::int64_t base =
                        (position & 2047) +
                        static_cast<std::int64_t>(kCachePadded) * (head + 8 * slot);
                    if (!sample_ok(k_got[base * 128 + pair], kx, 4.0e-2) ||
                        !sample_ok(k_got[base * 128 + pair + 64], ky, 4.0e-2) ||
                        !sample_ok(k_got[base * 128 + pair + 1], kx1, 4.0e-2) ||
                        !sample_ok(k_got[base * 128 + pair + 65], ky1, 4.0e-2)) {
                        std::cerr << "context K layer=" << layer << " column=" << column
                                  << " head=" << head << " pair=" << pair << ": actual k="
                                  << k_got[base * 128 + pair] << " reference=" << kx << '\n';
                        ++failures;
                    }
                }
                for (const std::int32_t dim : {0, 64, 127}) {
                    const std::int32_t row = head * 128 + dim;
                    double value           = 0.0;
                    for (std::int32_t input = 0; input < kHidden; ++input) {
                        value += quantized_weight::logical_weight_fp64(values[layer], row, input) *
                                 context[column * kHidden + input];
                    }
                    const std::int64_t base =
                        (position & 2047) +
                        static_cast<std::int64_t>(kCachePadded) * (head + 8 * slot);
                    if (!sample_ok(v_got[base * 128 + dim], value, 4.0e-2)) {
                        std::cerr << "context V layer=" << layer << " column=" << column
                                  << " head=" << head << " dim=" << dim
                                  << ": actual=" << v_got[base * 128 + dim]
                                  << " reference=" << value << '\n';
                        ++failures;
                    }
                }
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
    int failures = 0;
    std::cerr << "section: attn_input\n";
    failures += verify_attn_input();
    std::cerr << "section: conv\n";
    const PackedWeight conv_projection_kernel = nvfp4_weight(1280, kHidden, 521U);
    const PackedWeight attention_output       = nvfp4_weight(kHidden, 4096, 523U);
    const PackedWeight mlp_down               = nvfp4_weight(kHidden, 17408, 525U);
    std::cerr << "section: conv 2x1\n";
    failures += verify_dynamic_conv(conv_projection_kernel, attention_output, 2, 1, 701U);
    std::cerr << "section: conv 8x2\n";
    failures += verify_dynamic_conv(conv_projection_kernel, attention_output, 8, 2, 703U);
    std::cerr << "section: conv 4x3\n";
    failures += verify_dynamic_conv(conv_projection_kernel, mlp_down, 4, 3, 705U);
    std::cerr << "section: context\n";
    failures += verify_context_kv(1, 1, 801U, false);
    failures += verify_context_kv(16, 1, 803U, false);
    failures += verify_context_kv(3, 2, 805U, false);
    std::cerr << "section: context graph\n";
    failures += verify_context_kv(1, 1, 801U, true);
    failures += verify_context_kv(8, 1, 807U, true);
    failures += verify_context_kv(16, 1, 803U, true);
    failures += verify_context_kv(3, 2, 805U, true);
    failures += verify_context_kv(9, 2, 809U, true);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " dflash2 nvfp4 routes\n";
    return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "dflash2 nvfp4 routes: " << error.what() << '\n';
        return 1;
    }
}
