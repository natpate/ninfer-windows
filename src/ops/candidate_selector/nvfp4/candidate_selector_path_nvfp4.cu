// NVFP4 codebook route of candidate_selector_path: the [248320,256] codebooks are weight-only
// NVFP4 (packed E2M1 pairs with one E4M3 scale per 16-rank group and a payload divisor). The
// walk structure, draw, and lattice are shared with the BF16 family; only row access decodes.
// Successor rows stage raw codes and scales through cp_async and decode to BF16 after the wait;
// predecessor elements decode inline at their point of use.
#include "ops/candidate_selector/nvfp4/candidate_selector_path_nvfp4.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kCandidates = 16, kRank = 256;

constexpr int kCodeBytesPerRow  = kRank / 2;
constexpr int kScaleBytesPerRow = kRank / 16;

// The codebook scale plane follows the registered K16M128x4 blocked arrangement shared by every
// NVFP4 payload: 512-byte tiles of 128 rows x 4 consecutive groups.
__device__ __forceinline__ int codebook_scale_offset(int token, int group) {
    const int in_tile = (token % 32) * 16 + ((token % 128) / 32) * 4;
    return (token / 128) * 4 * 512 + (group / 4) * 512 + in_tile + group % 4;
}

struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    const std::uint8_t* predecessor_codes;
    const std::uint8_t* predecessor_scales;
    float predecessor_inverse_divisor;
    const std::uint8_t* successor_codes;
    const std::uint8_t* successor_scales;
    float successor_inverse_divisor;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    int steps;
};

struct alignas(16) SelectorShared {
    std::uint8_t successor_staged[kCandidates * kCodeBytesPerRow];
    std::uint8_t successor_staged_scales[kCandidates * kScaleBytesPerRow];
    __nv_bfloat16 successors[kCandidates * kRank];
    float product[kRank];
    float edge[kCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

__device__ __forceinline__ float decode_rank(const std::uint8_t* codes,
                                             const std::uint8_t* scales, int token, int rank) {
    const float2 pair = decode_nvfp4_e2m1x2(codes[token * kCodeBytesPerRow + (rank >> 1)]);
    const float code  = (rank & 1) != 0 ? pair.y : pair.x;
    return code * decode_nvfp4_e4m3(scales[codebook_scale_offset(token, rank >> 4)]);
}

// The probabilities written here are the same FP32 values consumed by the draw.
__device__ int draw_rank(float edge, float temperature, unsigned long long seed, int position,
                         float* q) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits = __ballot_sync(kFullWarpMask, lane < kCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kCandidates - 1;
}

__device__ void score_row(const DeviceArgs& a, int column, SelectorShared& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int token = lane == 0 ? a.ids[column * kCandidates + warp] : 0;
    token     = __shfl_sync(kFullWarpMask, token, 0);
    // Stage one packed successor row per warp: 128 code bytes (row-major) and the row's four
    // 4-byte scale groups, one per 512-byte K16M128x4 tile, gathered into natural group order.
    if (lane < kCodeBytesPerRow / 16) {
        cp_async<16, Cache::cg>(
            &shared.successor_staged[warp * kCodeBytesPerRow + lane * 16],
            a.successor_codes + static_cast<std::int64_t>(token) * kCodeBytesPerRow + lane * 16);
    }
    if (lane < 4) {
        const int in_tile = (token % 32) * 16 + ((token % 128) / 32) * 4;
        cp_async<4>(&shared.successor_staged_scales[warp * kScaleBytesPerRow + lane * 4],
                    a.successor_scales + (static_cast<std::int64_t>(token / 128) * 4 + lane) *
                                              512 +
                        in_tile);
    }
    cp_commit();
    // Publish the preceding draw before reading its token. Successor prefetch is independent.
    __syncthreads();
    if (tid < kRank) {
        shared.product[tid] =
            decode_rank(a.predecessor_codes, a.predecessor_scales, shared.predecessor, tid) *
            a.predecessor_inverse_divisor *
            __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kRank + tid]);
    }
    cp_wait<0>();
    __syncthreads();
    for (int item = tid; item < kCandidates * kRank; item += 512) {
        const int candidate = item / kRank, rank = item % kRank;
        const float2 pair   = decode_nvfp4_e2m1x2(
            shared.successor_staged[candidate * kCodeBytesPerRow + (rank >> 1)]);
        const float code = (rank & 1) != 0 ? pair.y : pair.x;
        const float scale =
            decode_nvfp4_e4m3(shared.successor_staged_scales[candidate * kScaleBytesPerRow +
                                                             (rank >> 4)]);
        shared.successors[item] =
            __float2bfloat16_rn(code * scale * a.successor_inverse_divisor);
    }
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
#pragma unroll
        for (int r = lane; r < kRank; r += 32)
            sum = fmaf(shared.product[r], __bfloat162float(shared.successors[c * kRank + r]), sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = a.unary[column * kCandidates + c] + sum;
    }
    __syncthreads();
}

__global__ __launch_bounds__(512, 1) void selector_walk_nvfp4_kernel(DeviceArgs a) {
    __shared__ SelectorShared shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor   = a.anchors[batch];
        shared.base_position = a.positions[batch];
        shared.temperature   = a.configs[batch].temperature;
        shared.seed          = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        score_row(a, column, shared);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed,
                                           shared.base_position + step, a.q + column * kCandidates);
            if (lane == 0) {
                shared.predecessor = a.ids[column * kCandidates + selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

__global__ __launch_bounds__(512, 2) void selector_lattice_nvfp4_kernel(DeviceArgs a, float* edges) {
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if (threadIdx.x == 0)
        shared.predecessor = step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
    score_row(a, column, shared);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

__global__ __launch_bounds__(32) void selector_lattice_walk_nvfp4_kernel(DeviceArgs a,
                                                                         const float* edges) {
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kCandidates
                ? edges[(static_cast<std::int64_t>(column) * kCandidates + predecessor_rank) *
                            kCandidates +
                        lane]
                : -CUDART_INF_F;
        int selected =
            draw_rank(edge, temperature, seed, position + step, a.q + column * kCandidates);
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kCandidates + selected];
    }
}

} // namespace

void candidate_selector_path_nvfp4_launch(SelectorRoute route, const Tensor& candidate_ids,
                                          const Tensor& unary_scores,
                                          const Tensor& projected_hidden, const Tensor& anchors,
                                          const Weight& predecessor_codebook,
                                          const Weight& successor_codebook,
                                          const Tensor& base_positions,
                                          const SamplingConfig* configs, Tensor& drafts,
                                          Tensor& proposal_q, const SelectorWorkspace& workspace,
                                          cudaStream_t stream) {
    const DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                          static_cast<const float*>(unary_scores.data),
                          static_cast<const __nv_bfloat16*>(projected_hidden.data),
                          static_cast<const std::int32_t*>(anchors.data),
                          static_cast<const std::uint8_t*>(predecessor_codebook.qdata),
                          static_cast<const std::uint8_t*>(predecessor_codebook.scales),
                          1.0F / predecessor_codebook.weight_scale_divisor,
                          static_cast<const std::uint8_t*>(successor_codebook.qdata),
                          static_cast<const std::uint8_t*>(successor_codebook.scales),
                          1.0F / successor_codebook.weight_scale_divisor,
                          static_cast<const std::int32_t*>(base_positions.data),
                          configs,
                          static_cast<std::int32_t*>(drafts.data),
                          static_cast<float*>(proposal_q.data),
                          candidate_ids.ne[1]};
    if (route == SelectorRoute::Direct) {
        selector_walk_nvfp4_kernel<<<candidate_ids.ne[2], 512, 0, stream>>>(args);
    } else {
        auto* edges = static_cast<float*>(workspace.edges.data);
        selector_lattice_nvfp4_kernel<<<dim3(args.steps * candidate_ids.ne[2], kCandidates), 512, 0,
                                        stream>>>(args, edges);
        CUDA_CHECK(cudaGetLastError());
        selector_lattice_walk_nvfp4_kernel<<<candidate_ids.ne[2], 32, 0, stream>>>(args, edges);
    }
    CUDA_CHECK(cudaGetLastError());
}

void candidate_selector_path_nvfp4_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                            const Tensor& projected_hidden, const Tensor& anchors,
                                            const Weight& predecessor_codebook,
                                            const Weight& successor_codebook,
                                            const Tensor& base_positions,
                                            const SamplingConfig* configs, Tensor& drafts,
                                            Tensor& proposal_q, WorkspaceArena& workspace,
                                            cudaStream_t stream) {
    const auto route = candidate_selector_path_route(candidate_ids.ne[1], candidate_ids.ne[2]);
    auto scope       = workspace.scope();
    const auto scratch =
        allocate_selector_workspace(workspace, route, candidate_ids.ne[1], candidate_ids.ne[2]);
    candidate_selector_path_nvfp4_launch(route, candidate_ids, unary_scores, projected_hidden,
                                         anchors, predecessor_codebook, successor_codebook,
                                         base_positions, configs, drafts, proposal_q, scratch,
                                         stream);
}

} // namespace ninfer::ops::detail
