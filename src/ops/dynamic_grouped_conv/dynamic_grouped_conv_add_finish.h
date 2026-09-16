#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Finish step shared by every projection weight format of linear_dynamic_grouped_conv_add:
// projected is the materialized BF16 [5120, width*batch] projection, and the kernel folds the
// two-tap dynamic convolution into the residual in place.
void dynamic_grouped_conv_add_finish_launch(const Tensor& projected, const Tensor& base_kernel,
                                            const Tensor& finish_delta, Tensor& residual,
                                            cudaStream_t stream);

} // namespace ninfer::ops::detail
