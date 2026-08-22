// ninfer::ops - BF16 small-T partial-kernel route. Split from the unified
// dispatcher so this kernel family compiles in its own translation unit.
#include "ops/launcher/gqa_attention.h"

#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k = cache.k_pages;
    Tensor& cache_v = cache.v_pages;
    // bf16 kernel uses only static smem (no dynamic staging).
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, MultiBatch,
                                                 Masked, CacheInput><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), input,
        static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
        static_cast<__nv_bfloat16*>(cache_v.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        invocation.valid_columns == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.valid_columns->data),
        invocation.table_rows == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(invocation.table_rows->data),
        cache.block_tables.ne[0], invocation.width, invocation.full_width, invocation.column_begin,
        logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

template <typename Geometry, typename CacheInput>
void gqa_small_t_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                              PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                              std::int32_t logical_capacity, std::int32_t splits,
                              Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                              cudaStream_t stream) {
    // BF16 keeps its row-tile warp count (2 warps at T=1, 4 warps above).
#define NINFER_GQA_SMALL_T_BF16_DISPATCH(TOKENS, WARPS)                                             \
    do {                                                                                           \
        const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {                          \
            launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS), MultiBatch, Masked>(               \
                q, input, pos, scale, cache, invocation, logical_capacity, splits, partial_acc,    \
                partial_m, partial_l, stream);                                                     \
        };                                                                                         \
        const bool masked = invocation.valid_columns != nullptr;                                   \
        if (invocation.batch_size == 1) {                                                          \
            if (masked) {                                                                          \
                launch_profile.template operator()<false, true>();                                 \
            } else {                                                                               \
                launch_profile.template operator()<false, false>();                                \
            }                                                                                      \
        } else if (masked) {                                                                       \
            launch_profile.template operator()<true, true>();                                      \
        } else {                                                                                   \
            launch_profile.template operator()<true, false>();                                     \
        }                                                                                          \
    } while (0)

    switch (invocation.width) {
    case 1:
        NINFER_GQA_SMALL_T_BF16_DISPATCH(1, 2);
        break;
    case 2:
        NINFER_GQA_SMALL_T_BF16_DISPATCH(2, 4);
        break;
    case 3:
        NINFER_GQA_SMALL_T_BF16_DISPATCH(3, 4);
        break;
    case 4:
        NINFER_GQA_SMALL_T_BF16_DISPATCH(4, 4);
        break;
    case 5:
        NINFER_GQA_SMALL_T_BF16_DISPATCH(5, 4);
        break;
    case 6:
        NINFER_GQA_SMALL_T_BF16_DISPATCH(6, 4);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
#undef NINFER_GQA_SMALL_T_BF16_DISPATCH
}

template void gqa_small_t_partial_bf16<Gqa27Geometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
template void gqa_small_t_partial_bf16<Gqa27Geometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
template void gqa_small_t_partial_bf16<Gqa35Geometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
    cudaStream_t);
template void gqa_small_t_partial_bf16<Gqa35Geometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

} // namespace ninfer::ops::detail
