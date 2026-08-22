// ninfer::ops - INT8 prompt/prefill route. Split from the unified dispatcher so
// this kernel family compiles in its own translation unit.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_attention_i8(const Tensor& q, const Tensor& positions, float scale,
                              const CacheView& cache, Metadata metadata, Tensor& out,
                              cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(gqa_attention_prefill_i8_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillI8SmemBytes);
    CUDA_CHECK(attr_i8);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillI8Br)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    const Tensor& cache_k_scale = cache.k_scale_pages;
    const Tensor& cache_v_scale = cache.v_scale_pages;
    gqa_attention_prefill_i8_kernel<Geometry, Metadata>
        <<<attention_grid, kGqaPrefillI8Threads, kGqaPrefillI8SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<std::int8_t*>(cache_k.data),
            static_cast<std::int8_t*>(cache_v.data), static_cast<const __half*>(cache_k_scale.data),
            static_cast<const __half*>(cache_v_scale.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_append_i8(const Tensor& k, const Tensor& v, const Tensor& positions,
                           CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    Tensor& cache_k_scale    = cache.k_scale_pages;
    Tensor& cache_v_scale    = cache.v_scale_pages;
    if (tokens >= 128 && Geometry::KVHeads == 2) {
        constexpr int kPageBlock     = 256;
        constexpr int kTokensPerTile = 8;
        const int max_tiles          = div_up(tokens + kTokensPerTile - 1, kTokensPerTile);
        const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                             static_cast<unsigned>(Geometry::KVHeads),
                             static_cast<unsigned>(kGqaKvQuantGroups));
        gqa_attention_prefill_fill_i8_page_kernel<Geometry, Metadata>
            <<<fill_grid, kPageBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::int8_t*>(cache_k.data), static_cast<std::int8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                tokens);
    } else {
        constexpr int kFillWarps     = 256 / 32;
        const std::int64_t fill_units =
            static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvQuantGroups;
        const int fill_grid =
            static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
        gqa_attention_prefill_fill_i8_kernel<Geometry, Metadata>
            <<<fill_grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(k.data),
                static_cast<const __nv_bfloat16*>(v.data),
                static_cast<const std::int32_t*>(positions.data), metadata,
                static_cast<std::int8_t*>(cache_k.data), static_cast<std::int8_t*>(cache_v.data),
                static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
                tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

#define NINFER_GQA_PREFILL_I8_INSTANTIATE(GEOMETRY, CACHE_VIEW, METADATA)                           \
    template void gqa_prefill_attention_i8<GEOMETRY, CACHE_VIEW, METADATA>(                         \
        const Tensor&, const Tensor&, float, const CACHE_VIEW&, METADATA, Tensor&, cudaStream_t);   \
    template void gqa_prefill_append_i8<GEOMETRY, CACHE_VIEW, METADATA>(                           \
        const Tensor&, const Tensor&, const Tensor&, CACHE_VIEW, METADATA, cudaStream_t);

NINFER_GQA_PREFILL_I8_INSTANTIATE(Gqa27Geometry, PagedKVLayerView, GqaPrefillDirectMetadata)
NINFER_GQA_PREFILL_I8_INSTANTIATE(Gqa35Geometry, PagedKVLayerView, GqaPrefillDirectMetadata)
NINFER_GQA_PREFILL_I8_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView,
                                  GqaPrefillBatchMetadata<false>)
NINFER_GQA_PREFILL_I8_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<true>)
NINFER_GQA_PREFILL_I8_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView,
                                  GqaPrefillBatchMetadata<false>)
NINFER_GQA_PREFILL_I8_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView,
                                  GqaPrefillBatchMetadata<true>)
#undef NINFER_GQA_PREFILL_I8_INSTANTIATE

} // namespace ninfer::ops::detail
