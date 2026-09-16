#pragma once

// Shared device-side pieces of context_kv_materialize: the packed-column mapping, the per-layer
// weight/cache view, the fused key head store, and the key post-processing kernel entry. Both
// the W8 and NVFP4 projection kernels consume them; the view's code/scale pointers are raw and
// interpreted per weight format.

#include "core/tensor.h"
#include "ninfer/ops/context_kv_materialize.h"
#include "ops/common/dflash_rope.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

constexpr int kContextKVRows    = 1024;
constexpr int kContextKVHeadDim = 128;

__device__ __forceinline__ int context_column(int column, int width, int prefix) {
    return width == prefix ? column : column / prefix * width + column % prefix;
}

union alignas(16) ContextKVBf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

__device__ __forceinline__ int swizzle_128(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

struct DeviceLayerView {
    const std::uint8_t* key_codes;
    const std::uint8_t* key_scales;
    const std::uint8_t* value_codes;
    const std::uint8_t* value_scales;
    float key_inverse_divisor;
    float value_inverse_divisor;
    const __nv_bfloat16* key_norm;
    __nv_bfloat16* cache_k;
    __half* cache_v;
    std::int32_t padded_capacity;
};

struct DeviceLayers {
    DeviceLayerView layer[kContextKVMaterializeLayers];
};

__device__ __forceinline__ void store_key_head(const float* input, DeviceLayerView layer,
                                               const int* positions, const int* slots, int column,
                                               int width, int head) {
    const int lane = threadIdx.x & 31;
    const int j    = lane * 2;
    float x0 = input[j], x1 = input[j + 1], y0 = input[j + 64], y1 = input[j + 65];
    float sum         = warp_reduce_sum(x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1);
    const float inverse = rsqrtf(__shfl_sync(0xffffffffU, sum, 0) / 128.0f + 1.e-6f);
    x0 *= inverse * __bfloat162float(layer.key_norm[j]);
    x1 *= inverse * __bfloat162float(layer.key_norm[j + 1]);
    y0 *= inverse * __bfloat162float(layer.key_norm[j + 64]);
    y1 *= inverse * __bfloat162float(layer.key_norm[j + 65]);
    float sin0, cos0, sin1, cos1;
    dflash_rope_sincos(positions, column, j, &sin0, &cos0);
    dflash_rope_sincos(positions, column, j + 1, &sin1, &cos1);
    const auto dst = 128LL * ((positions[column] & 2047) + (long long)layer.padded_capacity *
                                                               (head + 8 * slots[column / width]));
    auto* out      = reinterpret_cast<__nv_bfloat162*>(layer.cache_k + dst);
    out[lane]      = __floats2bfloat162_rn(x0 * cos0 - y0 * sin0, x1 * cos1 - y1 * sin1);
    out[lane + 32] = __floats2bfloat162_rn(y0 * cos0 + x0 * sin0, y1 * cos1 + x1 * sin1);
}

inline DeviceLayers make_device_layers(
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers) {
    DeviceLayers result{};
    for (int index = 0; index < static_cast<int>(kContextKVMaterializeLayers); ++index) {
        const ContextKVMaterializeLayerView& source = layers[static_cast<std::size_t>(index)];
        const auto inverse = [](const Weight& weight) {
            return weight.qtype == QType::NVFP4 ? 1.0F / weight.weight_scale_divisor : 1.0F;
        };
        result.layer[index] = {
            static_cast<const std::uint8_t*>(source.key_weight.qdata),
            static_cast<const std::uint8_t*>(source.key_weight.scales),
            static_cast<const std::uint8_t*>(source.value_weight.qdata),
            static_cast<const std::uint8_t*>(source.value_weight.scales),
            inverse(source.key_weight),
            inverse(source.value_weight),
            static_cast<const __nv_bfloat16*>(source.key_norm_weight.data),
            static_cast<__nv_bfloat16*>(source.cache.k.data),
            static_cast<__half*>(source.cache.v.data),
            static_cast<std::int32_t>(source.cache.padded_capacity),
        };
    }
    return result;
}

void context_kv_key_post_launch(const Tensor& key_scratch, const Tensor& positions,
                                const Tensor& counts, const Tensor& state_slots,
                                const DeviceLayers& layers, std::int32_t batch_size,
                                std::int32_t width,
                                ContextKVMaterializeExecutionEnvelope envelope, cudaStream_t stream);

} // namespace ninfer::ops::detail
