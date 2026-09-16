#include "ops/dynamic_grouped_conv/nvfp4/nvfp4_dynamic_grouped_conv_prepare_plan.h"

#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden = 5120, kGroups = 320, kCoefficientRows = 1280;

// The BF16 route keeps the projected coefficients in FP32 through the conv application; this
// route reads them from the BF16 dynamic matrix the generic A16 linear materialized. The extra
// coefficient rounding belongs to the NVFP4 route's implementation profile.
template <int Capacity>
__global__ __launch_bounds__(Capacity * 16, 4) void nvfp4_dynamic_grouped_conv_prepare_finish_kernel(
    const __nv_bfloat16* base, const __nv_bfloat16* dynamic, __nv_bfloat16* prepared,
    __nv_bfloat16* finish, int width, int batch_size) {
    __shared__ float projected[4][Capacity];
    __shared__ float normalized[Capacity][16];
    const int tid = threadIdx.x, group = blockIdx.x, batch = blockIdx.y;
    if (tid < 4 * Capacity && tid % Capacity < width) {
        const int coefficient = tid / Capacity, position = tid % Capacity;
        const int row          = coefficient * kGroups + group;
        projected[coefficient][position] = __bfloat162float(
            dynamic[(static_cast<std::int64_t>(batch) * width + position) * kCoefficientRows + row]);
    }
    const int position = tid / 16, channel = tid % 16, hidden = group * 16 + channel;
    const std::int64_t offset = static_cast<std::int64_t>(batch * width + position) * kHidden + hidden;
    if (position < width) { normalized[position][channel] = __bfloat162float(prepared[offset]); }
    __syncthreads();
    if (tid < 2 * Capacity && tid % Capacity < width) {
        const int tap = tid / Capacity, pos = tid % Capacity;
        finish[((batch * width + pos) * 2 + tap) * kGroups + group] =
            __float2bfloat16_rn(projected[2 + tap][pos]);
    }
    if (position < width) {
        float value = (__bfloat162float(base[hidden]) + projected[0][position]) *
                      normalized[position][channel];
        if (position > 0)
            value = fmaf(__bfloat162float(base[kHidden + hidden]) + projected[1][position],
                         normalized[position - 1][channel], value);
        prepared[offset] = __float2bfloat16_rn(value);
    }
}

void launch_finish(const Tensor& base, const Tensor& dynamic, Tensor& prepared, Tensor& finish,
                   cudaStream_t stream) {
    const dim3 grid(kGroups, prepared.ne[2]);
    if (prepared.ne[1] <= 8) {
        nvfp4_dynamic_grouped_conv_prepare_finish_kernel<8>
            <<<grid, 8 * 16, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(base.data),
                static_cast<const __nv_bfloat16*>(dynamic.data),
                static_cast<__nv_bfloat16*>(prepared.data),
                static_cast<__nv_bfloat16*>(finish.data), prepared.ne[1], prepared.ne[2]);
    } else {
        nvfp4_dynamic_grouped_conv_prepare_finish_kernel<16>
            <<<grid, 16 * 16, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(base.data),
                static_cast<const __nv_bfloat16*>(dynamic.data),
                static_cast<__nv_bfloat16*>(prepared.data),
                static_cast<__nv_bfloat16*>(finish.data), prepared.ne[1], prepared.ne[2]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm,
                                                float eps, const Tensor& base,
                                                const Weight& kernel_projection, Tensor& prepared,
                                                Tensor& finish_delta, WorkspaceArena& workspace,
                                                cudaStream_t stream) {
    auto scope       = workspace.scope();
    const int tokens = residual.ne[1] * residual.ne[2];
    Tensor dynamic   = workspace.alloc(DType::BF16, {kCoefficientRows, tokens});
    rmsnorm(residual, norm, eps, false, prepared, stream);
    linear(prepared.view({kHidden, tokens}), kernel_projection, dynamic, stream);
    launch_finish(base, dynamic, prepared, finish_delta, stream);
}

void nvfp4_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& projection,
                                                    const Tensor& base_kernel,
                                                    const Tensor& finish_delta, Tensor& residual,
                                                    WorkspaceArena& workspace,
                                                    cudaStream_t stream) {
    auto scope       = workspace.scope();
    const int tokens = x.ne[1] * x.ne[2];
    Tensor projected = workspace.alloc(DType::BF16, {5120, tokens});
    linear(x.view({x.ne[0], tokens}), projection, projected, stream);
    dynamic_grouped_conv_add_finish_launch(projected, base_kernel, finish_delta, residual,
                                           stream);
}

} // namespace ninfer::ops::detail
