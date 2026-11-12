// Modified by JoaFoschiatti for Cinference sm_86: Q5 K-split MMA LinearAdd route.

#include "core/weight.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct Q5KSplitResidualEpilogue {
    __nv_bfloat16* residual;
    std::int32_t rows;

    __device__ __forceinline__ void add(int row, int col, float value, int live) const {
        if (col < live) {
            __nv_bfloat16* destination = residual + static_cast<std::int64_t>(col) * rows + row;
            *destination = __float2bfloat16_rn(__bfloat162float(*destination) + value);
        }
    }

    __device__ __forceinline__ void operator()(int row, int col, float4 value, int live) const {
        add(row, col, value.x, live);
        add(row + 8, col, value.z, live);
        add(row, col + 1, value.y, live);
        add(row + 8, col + 1, value.w, live);
    }
};

template <int InputRows, int TileCols>
void launch(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    using Geometry = Q5LinearGeometry<5120, InputRows>;
    q5_ksplit_mma_kernel<Geometry, TileCols, TileCols, Q5KSplitResidualEpilogue>
        <<<5120 / Q5KSplitMmaSchedule::kRowsPerCta, Q5KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            Q5KSplitResidualEpilogue{static_cast<__nv_bfloat16*>(residual_out.data), 5120},
            x.ne[1]);
}

template <int InputRows>
void launch_cols(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    if (x.ne[1] <= 8) {
        launch<InputRows, 8>(x, w, residual_out, stream);
    } else if (x.ne[1] <= 16) {
        launch<InputRows, 16>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add K-split MMA: T must be in [1,16]");
    }
}

} // namespace

void q5_linear_add_ksplit_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    if (residual_out.ne[0] != 5120 || w.padded_shape[1] != w.k) {
        throw std::invalid_argument("q5 linear_add K-split MMA: unsupported shape");
    }
    if (w.k == 6144) {
        launch_cols<6144>(x, w, residual_out, stream);
    } else if (w.k == 17408) {
        launch_cols<17408>(x, w, residual_out, stream);
    } else {
        throw std::invalid_argument("q5 linear_add K-split MMA: unsupported exact K");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
