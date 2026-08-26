// ninfer::ops - split-KV GQA small-T dispatcher: host split policy, dtype route
// selection, and the split reducer. The per-dtype partial kernels live in
// gqa_attention_decode_{bf16,i8}.cu.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t gqa_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::DecodeSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::DecodeSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::DecodeSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::DecodeSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::DecodeSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::DecodeSplitScale); }

    return (splits < Geometry::DecodeSplits) ? splits : Geometry::DecodeSplits;
}

template <typename Geometry>
std::int32_t gqa_small_t_split_count(std::int32_t window, std::int32_t tokens, DType kv_dtype) {
    // A 64-key default split just above a 32-key boundary makes the partial
    // kernel execute a nearly empty second tile. These short ranges instead
    // launch one 32-key tile per split; the larger CTAs keep the small grid busy.
    if (kv_dtype == DType::I8 && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::DecodeSplitScale);
    }
    if (kv_dtype == DType::I8 && tokens == 6 && window > 128 && window <= 160) {
        return div_up(window, 24 / Geometry::DecodeSplitScale);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (kv_dtype == DType::I8 && tokens == 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::DecodeSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::DecodeSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return gqa_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t gqa_small_t_launch_capacity(GqaExecutionEnvelope envelope, std::int32_t tokens,
                                         DType dtype) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits =
            gqa_small_t_split_count<Geometry>(static_cast<std::int32_t>(window), tokens, dtype);
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

PagedKVBatchLayerView single_row_batch_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .dtype         = cache.dtype,
        .quant_group   = cache.quant_group,
        .packed_v      = cache.packed_v,
        .rotate_k      = cache.rotate_k,
        .rotate_v      = cache.rotate_v,
        .packed_k      = cache.packed_k,
        .e8_lattice    = cache.e8_lattice,
        .e8_root       = cache.e8_root,
    };
}

} // namespace

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 6 || (cache_dtype != DType::BF16 && cache_dtype != DType::I8) ||
        envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys) {
        throw std::invalid_argument("gqa_attention split capacity: invalid profile");
    }
    if (q_heads == Gqa27Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa27Geometry>(envelope, tokens, cache_dtype);
    }
    if (q_heads == Gqa35Geometry::QHeads) {
        return gqa_small_t_launch_capacity<Gqa35Geometry>(envelope, tokens, cache_dtype);
    }
    throw std::invalid_argument("gqa_attention split capacity: unsupported Q-head count");
}

template <typename Geometry, typename CacheInput>
void gqa_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                      float scale, PagedKVBatchLayerView cache,
                                      const GqaSmallTInvocation& invocation,
                                      GqaExecutionEnvelope envelope,
                                      Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                      Tensor& out,
                                      cudaStream_t stream) {
    const auto logical_capacity      = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits =
        gqa_small_t_launch_capacity<Geometry>(envelope, invocation.width, cache.dtype);

    if (invocation.width < 1 || invocation.width > 6) {
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry and its compressed-KV packing inside the i8 route.
    if (cache.dtype == DType::I8) {
        gqa_small_t_partial_i8<Geometry, CacheInput>(
            q, input, pos, scale, cache, invocation, logical_capacity, implementation_window,
            splits, partial_acc, partial_m, partial_l, stream);
    } else {
        gqa_small_t_partial_bf16<Geometry, CacheInput>(q, input, pos, scale, cache, invocation,
                                                       logical_capacity, splits, partial_acc,
                                                       partial_m, partial_l, stream);
    }

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(Geometry::QHeads, div_up(kGqaHeadDim, kDChunk),
                           invocation.width * invocation.batch_size);
    const auto launch_reduce = [&]<bool Int8, bool MultiBatch, bool Masked, bool Offset>() {
        gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk, Int8, MultiBatch, Masked,
                                                   Offset>
            <<<reduce_grid, kReduceBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.width, invocation.full_width, invocation.column_begin,
                invocation.batch_size, splits, static_cast<__nv_bfloat16*>(out.data));
    };
    const bool masked         = invocation.valid_columns != nullptr;
    const auto launch_profile = [&]<bool Int8, bool MultiBatch, bool Masked>() {
        if (invocation.column_begin == 0) {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, false>();
        } else {
            launch_reduce.template operator()<Int8, MultiBatch, Masked, true>();
        }
    };
    const auto launch_for_dtype = [&]<bool Int8>() {
        if (invocation.batch_size == 1) {
            if (masked) {
                launch_profile.template operator()<Int8, false, true>();
            } else {
                launch_profile.template operator()<Int8, false, false>();
            }
        } else if (masked) {
            launch_profile.template operator()<Int8, true, true>();
        } else {
            launch_profile.template operator()<Int8, true, false>();
        }
    };
    if (cache.dtype == DType::I8) {
        launch_for_dtype.template operator()<true>();
    } else {
        launch_for_dtype.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
    if (cache.rotate_v) {
        const int units = invocation.batch_size * invocation.width * Geometry::QHeads *
                          kGqaKvQuantGroups;
        gqa_kv_inverse_rotate_output_kernel<Geometry::QHeads><<<units, 32, 0, stream>>>(
            static_cast<__nv_bfloat16*>(out.data), invocation.width, invocation.full_width,
            invocation.column_begin,
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data));
        CUDA_CHECK(cudaGetLastError());
    }
}

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, const Tensor& valid_columns,
                                  const Tensor& table_rows, float scale,
                                  PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                  std::int32_t column_begin, std::int32_t width,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream) {
    const GqaAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                               static_cast<const __nv_bfloat16*>(v.data)};
    const GqaSmallTInvocation invocation{
        .valid_columns = valid_columns.data == nullptr ? nullptr : &valid_columns,
        .table_rows    = &table_rows,
        .full_width    = q.ne[2],
        .column_begin  = column_begin,
        .width         = width,
        .batch_size    = q.ne[3],
    };
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, cache, invocation,
                                                        envelope, partial_acc, partial_m, partial_l,
                                                        out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream) {
    const GqaCachedInput input{};
    const GqaSmallTInvocation invocation{
        .valid_columns = nullptr,
        .table_rows    = nullptr,
        .full_width    = q.ne[2],
        .column_begin  = 0,
        .width         = q.ne[2],
        .batch_size    = 1,
    };
    const PagedKVBatchLayerView batch_cache = single_row_batch_view(cache);
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(q, input, pos, scale, batch_cache,
                                                        invocation, envelope, partial_acc,
                                                        partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, batch_cache, invocation,
                                                    envelope, partial_acc, partial_m, partial_l,
                                                    out, stream);
}

} // namespace ninfer::ops::detail
