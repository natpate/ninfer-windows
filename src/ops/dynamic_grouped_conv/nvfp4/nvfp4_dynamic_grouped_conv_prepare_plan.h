#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// NVFP4 route of rmsnorm_dynamic_grouped_conv_prepare: the kernel_projection parent is the
// weight-only NVFP4 [1280,5120] matrix. The route materializes the dynamic coefficients in BF16
// through the generic A16 linear and then applies the shared conv finish, so its transient
// footprint (1280 x W x B BF16) stays below the split-K capacity the BF16 route reports.
void nvfp4_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm,
                                                float eps, const Tensor& base,
                                                const Weight& kernel_projection, Tensor& prepared,
                                                Tensor& finish_delta, WorkspaceArena& workspace,
                                                cudaStream_t stream);

// NVFP4 route of linear_dynamic_grouped_conv_add: the projection parent is the weight-only NVFP4
// [5120,C] matrix (C in {4096,17408}). The route materializes the projection in BF16 through the
// generic A16 linear and then applies the shared format-neutral finish; its transient footprint
// matches the W8 route exactly.
void nvfp4_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& projection,
                                                    const Tensor& base_kernel,
                                                    const Tensor& finish_delta, Tensor& residual,
                                                    WorkspaceArena& workspace,
                                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
