#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream);

// Three-output DFlash2 route: the weight-only NVFP4 [6144,5120] parent writing q [4096,T],
// k [1024,T], and v [1024,T] directly at every positive T (32-token chunks above the small-T
// family). No transient workspace.
void nvfp4_dflash2_attn_input(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                              Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
