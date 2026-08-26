// ninfer::ops - BF16 prompt/prefill route. Split from the unified dispatcher so
// this kernel family compiles in its own translation unit.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_attention_bf16(const Tensor& q, const Tensor& positions, float scale,
                                const CacheView& cache, Metadata metadata, Tensor& out,
                                cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(gqa_attention_prefill_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillSmemBytes);
    CUDA_CHECK(attr_bf16);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    gqa_attention_prefill_bf16_kernel<Geometry, Metadata>
        <<<attention_grid, kGqaPrefillThreads, kGqaPrefillSmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(cache_k.data),
            static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_prefill_append_bf16(const Tensor& k, const Tensor& v, const Tensor& positions,
                             CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    constexpr int kBlock           = Geometry::KVHeads == 4 ? 128 : 96;
    constexpr int kFillVecElems    = 8;
    const std::int64_t kv_elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                     (kGqaPrefillHeadDim / kFillVecElems);
    const int fill_grid =
        static_cast<int>(div_up(kv_elements, static_cast<std::int64_t>(kBlock)));
    gqa_attention_prefill_fill_bf16_kernel<Geometry, Metadata>
        <<<fill_grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(k.data),
                                           static_cast<const __nv_bfloat16*>(v.data),
                                           static_cast<const std::int32_t*>(positions.data),
                                           metadata, static_cast<__nv_bfloat16*>(cache_k.data),
                                           static_cast<__nv_bfloat16*>(cache_v.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

#define NINFER_GQA_PREFILL_BF16_INSTANTIATE(GEOMETRY, CACHE_VIEW, METADATA)                         \
    template void gqa_prefill_attention_bf16<GEOMETRY, CACHE_VIEW, METADATA>(                       \
        const Tensor&, const Tensor&, float, const CACHE_VIEW&, METADATA, Tensor&, cudaStream_t);   \
    template void gqa_prefill_append_bf16<GEOMETRY, CACHE_VIEW, METADATA>(                          \
        const Tensor&, const Tensor&, const Tensor&, CACHE_VIEW, METADATA, cudaStream_t);

NINFER_GQA_PREFILL_BF16_INSTANTIATE(Gqa27Geometry, PagedKVLayerView, GqaPrefillDirectMetadata)
NINFER_GQA_PREFILL_BF16_INSTANTIATE(Gqa35Geometry, PagedKVLayerView, GqaPrefillDirectMetadata)
NINFER_GQA_PREFILL_BF16_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView,
                                    GqaPrefillBatchMetadata<false>)
NINFER_GQA_PREFILL_BF16_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView,
                                    GqaPrefillBatchMetadata<true>)
NINFER_GQA_PREFILL_BF16_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView,
                                    GqaPrefillBatchMetadata<false>)
NINFER_GQA_PREFILL_BF16_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView,
                                    GqaPrefillBatchMetadata<true>)
#undef NINFER_GQA_PREFILL_BF16_INSTANTIATE

} // namespace ninfer::ops::detail
