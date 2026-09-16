// Format-neutral key post-processing of context_kv_materialize: RMSNorm + DFlash rope over the
// FP32 key scratch and the store into the cyclic cache. Shared by every weight format.
#include "ops/context_kv_materialize/context_kv_common.cuh"

#include "core/device.h"

namespace ninfer::ops::detail {
namespace {

__global__ __launch_bounds__(256) void context_kv_key_post_kernel(
    const float* __restrict__ key_scratch, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ state_slots,
    DeviceLayers layers, std::int32_t batch_size, std::int32_t width, std::int32_t min_count,
    std::int32_t max_count) {
    const int packed_column   = static_cast<int>(blockIdx.x);
    const int physical_column = context_column(packed_column, width, max_count);
    const int layer_index     = static_cast<int>(blockIdx.y);
    const int batch           = physical_column / width;
    const int local           = physical_column % width;
    const int count           = counts[batch];
    if (count < min_count || count > max_count || local >= count) return;
    const auto layer   = layers.layer[layer_index];
    const int head     = threadIdx.x >> 5;
    const float* input = key_scratch +
                         kContextKVRows * (packed_column + max_count * batch_size * layer_index) +
                         head * kContextKVHeadDim;
    store_key_head(input, layer, positions, state_slots, physical_column, width, head);
}

} // namespace

void context_kv_key_post_launch(const Tensor& key_scratch, const Tensor& positions,
                                const Tensor& counts, const Tensor& state_slots,
                                const DeviceLayers& layers, std::int32_t batch_size,
                                std::int32_t width,
                                ContextKVMaterializeExecutionEnvelope envelope,
                                cudaStream_t stream) {
    context_kv_key_post_kernel<<<dim3(envelope.max_count * batch_size,
                                       static_cast<unsigned>(kContextKVMaterializeLayers)),
                                 256, 0, stream>>>(
        static_cast<const float*>(key_scratch.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(counts.data), static_cast<const int*>(state_slots.data), layers,
        batch_size, width, static_cast<std::int32_t>(envelope.min_count),
        static_cast<std::int32_t>(envelope.max_count));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
