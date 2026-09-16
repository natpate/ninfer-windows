#pragma once

#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

namespace ninfer::ops::detail {

// Weight-only NVFP4 codebook route of the selector walk; see the BF16 family for the route
// structure and the shared SelectorWorkspace contract.
void candidate_selector_path_nvfp4_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                            const Tensor& projected_hidden, const Tensor& anchors,
                                            const Weight& predecessor_codebook,
                                            const Weight& successor_codebook,
                                            const Tensor& base_positions,
                                            const SamplingConfig* configs, Tensor& drafts,
                                            Tensor& proposal_q, WorkspaceArena& workspace,
                                            cudaStream_t stream);

} // namespace ninfer::ops::detail
