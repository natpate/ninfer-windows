// Format-neutral finish step of linear_dynamic_grouped_conv_add: applies the two-tap dynamic
// convolution with the base kernel and finish delta to the materialized projection and
// accumulates into the residual. Shared by every projection weight format.
#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.h"

#include "core/device.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 5120, kGroups = 320;

__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    const int index = col * kRows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * kRows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * kRows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    const int index = col * kRows + row;
    finish_value(row, col, width, __bfloat162float(projected[index]),
                 col % width ? __bfloat162float(projected[index - kRows]) : 0.0f, base, delta,
                 residual);
}

} // namespace

void dynamic_grouped_conv_add_finish_launch(const Tensor& projected, const Tensor& base_kernel,
                                            const Tensor& finish_delta, Tensor& residual,
                                            cudaStream_t stream) {
    const int tokens = residual.ne[1] * residual.ne[2];
    const dim3 grid((kRows + 255) / 256, tokens);
    finish_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const __nv_bfloat16*>(base_kernel.data),
        static_cast<const __nv_bfloat16*>(finish_delta.data),
        static_cast<__nv_bfloat16*>(residual.data), residual.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
