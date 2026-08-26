// ninfer::ops - INT8 small-T partial-kernel route. Split from the unified
// dispatcher so this kernel family compiles in its own translation unit. The
// compressed-KV packing/codec variant (packed_v / packed_k / e8_lattice /
// e8_root) is selected here from the cache view flags.
#include "ops/launcher/gqa_attention.h"

#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, int TokenTile, bool PackedV, bool RotateK, bool RotateV,
          bool PackedK, bool E8Lattice, bool E8Root, bool MultiBatch, bool Masked,
          typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                          std::int32_t logical_capacity, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    const auto launch =
        [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kGqaHeadDim) : 0ULL;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     PackedV, RotateK, RotateV, PackedK,
                                                     E8Lattice, E8Root, MultiBatch, Masked, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                             KeyBlock, DynamicArena, PackedV, RotateK, RotateV,
                                             PackedK, E8Lattice, E8Root, MultiBatch, Masked, CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::uint8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    if constexpr (TokenTile == 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

template <typename Geometry, typename CacheInput>
void gqa_small_t_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                            std::int32_t logical_capacity, std::int32_t implementation_window,
                            std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                            Tensor& partial_l, cudaStream_t stream) {
#define NINFER_GQA_SMALL_T_I8_DISPATCH(TOKENS)                                                      \
    do {                                                                                           \
        const auto launch_packing = [&]<bool PackedV, bool RotateK, bool RotateV, bool PackedK,    \
                                      bool E8Lattice, bool E8Root>() {                             \
            const auto launch_profile = [&]<bool MultiBatch, bool Masked>() {                      \
                launch_tc_partial_i8<Geometry, (TOKENS), PackedV, RotateK, RotateV, PackedK,       \
                                     E8Lattice, E8Root, MultiBatch, Masked>(                       \
                    q, input, pos, scale, cache, invocation, logical_capacity,                     \
                    implementation_window, splits, partial_acc, partial_m, partial_l, stream);     \
            };                                                                                     \
            const bool masked = invocation.valid_columns != nullptr;                               \
            if (invocation.batch_size == 1) {                                                      \
                if (masked) {                                                                      \
                    launch_profile.template operator()<false, true>();                             \
                } else {                                                                           \
                    launch_profile.template operator()<false, false>();                            \
                }                                                                                  \
            } else if (masked) {                                                                   \
                launch_profile.template operator()<true, true>();                                  \
            } else {                                                                               \
                launch_profile.template operator()<true, false>();                                 \
            }                                                                                      \
        };                                                                                         \
        if (cache.e8_root) {                                                                       \
            launch_packing.template operator()<true, true, true, false, false, true>();            \
        } else if (cache.e8_lattice) {                                                             \
            launch_packing.template operator()<true, true, true, true, true, false>();             \
        } else if (cache.packed_k) {                                                               \
            launch_packing.template operator()<true, true, true, true, false, false>();            \
        } else if (cache.packed_v) {                                                               \
            launch_packing.template operator()<true, true, true, false, false, false>();           \
        } else {                                                                                   \
            launch_packing.template operator()<false, false, false, false, false, false>();        \
        }                                                                                          \
    } while (0)

    switch (invocation.width) {
    case 1:
        NINFER_GQA_SMALL_T_I8_DISPATCH(1);
        break;
    case 2:
        NINFER_GQA_SMALL_T_I8_DISPATCH(2);
        break;
    case 3:
        NINFER_GQA_SMALL_T_I8_DISPATCH(3);
        break;
    case 4:
        NINFER_GQA_SMALL_T_I8_DISPATCH(4);
        break;
    case 5:
        NINFER_GQA_SMALL_T_I8_DISPATCH(5);
        break;
    case 6:
        NINFER_GQA_SMALL_T_I8_DISPATCH(6);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
#undef NINFER_GQA_SMALL_T_I8_DISPATCH
}

template void gqa_small_t_partial_i8<Gqa27Geometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&,
    Tensor&, cudaStream_t);
template void gqa_small_t_partial_i8<Gqa27Geometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&,
    Tensor&, cudaStream_t);
template void gqa_small_t_partial_i8<Gqa35Geometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&,
    Tensor&, cudaStream_t);
template void gqa_small_t_partial_i8<Gqa35Geometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&,
    Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
