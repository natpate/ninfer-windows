// ninfer::ops - gqa_attention prompt-scale dispatcher: geometry, metadata, and
// dtype route selection. The per-dtype kernels live in
// gqa_attention_prefill_{bf16,i8}.cu; this TU instantiates no fat kernels.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_append_route(const Tensor& k, const Tensor& v, const Tensor& positions,
                              CacheView cache, Metadata metadata, cudaStream_t stream) {
    if (cache.dtype == DType::I8) {
        gqa_prefill_append_i8<Geometry, CacheView, Metadata>(k, v, positions, cache, metadata,
                                                             stream);
    } else {
        gqa_prefill_append_bf16<Geometry, CacheView, Metadata>(k, v, positions, cache, metadata,
                                                               stream);
    }
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_attention_route(const Tensor& q, const Tensor& positions, float scale,
                                 const CacheView& cache, Metadata metadata, Tensor& out,
                                 cudaStream_t stream) {
    if (cache.dtype == DType::I8) {
        gqa_prefill_attention_i8<Geometry, CacheView, Metadata>(q, positions, scale, cache,
                                                                metadata, out, stream);
    } else {
        gqa_prefill_attention_bf16<Geometry, CacheView, Metadata>(q, positions, scale, cache,
                                                                  metadata, out, stream);
    }
    // Compressed-KV formats store V rotated; un-rotate the output rows once
    // after attention (plain bf16/int8 caches never set rotate_v).
    if (cache.rotate_v) {
        const auto tokens = static_cast<std::int32_t>(q.ne[2]);
        gqa_kv_inverse_rotate_output_kernel<Geometry::QHeads>
            <<<tokens * Geometry::QHeads * kGqaKvQuantGroups, 32, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data), tokens, tokens, 0, nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_prefill_attention_route<Gqa27Geometry>(q, positions, scale, cache, metadata, out,
                                                   stream);
        return;
    }
    gqa_prefill_attention_route<Gqa35Geometry>(q, positions, scale, cache, metadata, out, stream);
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (k.ne[1] == Gqa27Geometry::KVHeads) {
        gqa_prefill_append_route<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
        return;
    }
    gqa_prefill_append_route<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == Gqa27Geometry::QHeads) {
            gqa_prefill_append_route<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
            gqa_prefill_attention_route<Gqa27Geometry>(q, positions, scale, cache, metadata, out,
                                                       stream);
            return;
        }
        gqa_prefill_append_route<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
        gqa_prefill_attention_route<Gqa35Geometry>(q, positions, scale, cache, metadata, out,
                                                   stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
