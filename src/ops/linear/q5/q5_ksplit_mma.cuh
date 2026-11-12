// Modified by JoaFoschiatti for Cinference sm_86: Q5 form of q4_ksplit_mma.cuh.

#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int OutputRows, int InputRows>
struct Q5LinearGeometry {
    static constexpr int kOutputRows   = OutputRows;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

struct Q5KSplitMmaSchedule {
    static constexpr int kKWarps       = 8;
    static constexpr int kThreads      = kKWarps * 32;
    static constexpr int kTileKPerWarp = 64;
    static constexpr int kGroupK       = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta   = 16;
    static constexpr int kHighRowBytes = kGroupK / 8 + 16;
};

__device__ __forceinline__ int q5_ksplit_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int q5_ksplit_code_offset(int row, int byte) {
    return (((byte >> 4) ^ (row & 7)) << 4) | (byte & 15);
}

union Q5KSplitBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q5_ksplit_bf16_pair(std::uint8_t packed, unsigned high) {
    const unsigned value = packed;
    const unsigned lo    = (value & 0x0fu) | ((high & 1u) << 4);
    const unsigned hi    = (value >> 4) | ((high & 2u) << 3);
    Q5KSplitBf16PairBits biased;
    biased.bits = (lo | (hi << 16)) ^ 0x43104310u;
    Q5KSplitBf16PairBits bias;
    bias.bits = 0x43104310u;
    Q5KSplitBf16PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

template <class Geometry, int TileCols, int ActiveCols, class Epilogue, int Stages = 2>
__launch_bounds__(Q5KSplitMmaSchedule::kThreads, 4) __global__
    void q5_ksplit_mma_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ high,
                              const std::uint8_t* __restrict__ scales, Epilogue epilogue,
                              int columns) {
    using Schedule              = Q5KSplitMmaSchedule;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRows         = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kHighBytes    = kHidden / 8;
    constexpr int kNt           = TileCols / 8;
    constexpr int kCodeChunks   = kGroupK / 32;
    constexpr int kHighChunks   = kGroupK / 128;
    static_assert(TileCols == 8 || TileCols == 16);
    static_assert(ActiveCols >= 1 && ActiveCols <= TileCols);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(Stages >= 1 && Stages <= 3);

    union SharedStorage {
        struct {
            std::uint8_t codes[Stages][kRows][kGroupK / 2];
            std::uint8_t high[Stages][kRows][Schedule::kHighRowBytes];
            __nv_bfloat16 activations[Stages][kWarps][TileCols * kTileK];
            std::uint16_t scales[Stages][kRows][kWarps];
        } staging;

        float partial[kWarps * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int k_split  = warp;
    const int row0     = static_cast<int>(blockIdx.x) * kRows;
    const int live     = columns;

    const auto stage_x = [&](int stage, int group_k0) {
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col    = item / (kTileK / 8);
            const int k8     = item - col * (kTileK / 8);
            auto* dst        = &shared.staging.activations[stage][warp][col * kTileK +
                                                                 q5_ksplit_swizzle_64(col, k8 * 8)];
            const int source = col < live ? col : 0;
            cp_async_zfill<16>(dst,
                               &x[static_cast<std::int64_t>(source) * kHidden + group_k0 +
                                  warp * kTileK + k8 * 8],
                               col < live ? 16 : 0);
        }
    };

    const auto stage_weight = [&](int stage, int group_k0) {
        for (int task = tid; task < kRows * kCodeChunks; task += kWarps * 32) {
            const int row   = task / kCodeChunks;
            const int chunk = task - row * kCodeChunks;
            cp_async<16, Cache::cg>(
                &shared.staging.codes[stage][row][q5_ksplit_code_offset(row, chunk * 16)],
                codes + static_cast<std::int64_t>(row0 + row) * kCodeRowBytes + group_k0 / 2 +
                    chunk * 16);
        }
        for (int task = tid; task < kRows * kHighChunks; task += kWarps * 32) {
            const int row   = task / kHighChunks;
            const int chunk = task - row * kHighChunks;
            cp_async<16, Cache::cg>(&shared.staging.high[stage][row][chunk * 16],
                                    high + static_cast<std::int64_t>(row0 + row) * kHighBytes +
                                        group_k0 / 8 + chunk * 16);
        }
        for (int row = tid; row < kRows; row += kWarps * 32) {
            cp_async<kWarps * 2>(&shared.staging.scales[stage][row][0],
                                 scales + (static_cast<std::int64_t>(row0 + row) *
                                               Geometry::kGroupsPerRow +
                                           group_k0 / 64) *
                                              2);
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_koff = k_split * kTileK;
    float acc[kNt][4]   = {};

    const auto multiply_group = [&](int stage) {
        const auto& code_shared  = shared.staging.codes[stage];
        const auto& high_shared  = shared.staging.high[stage];
        const auto& x_shared     = shared.staging.activations[stage];
        const auto& scale_shared = shared.staging.scales[stage];
        float group_acc[kNt][4]  = {};

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned bf[kNt][2];
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int br = nt * 8 + b_rin;
                ldmatrix_x2(bf[nt][0], bf[nt][1],
                            smem_addr(&x_shared[k_split][br * kTileK + q5_ksplit_swizzle_64(
                                                                           br, ks * 16 + b_koff)]));
            }
            const int byte_col = warp_koff / 2 + ks * 8 + lid;
            const int offset   = q5_ksplit_code_offset(gid, byte_col);
            const int shift    = (byte_col & 3) * 2;
            const unsigned h0  = high_shared[gid][byte_col >> 2] >> shift;
            const unsigned h1  = high_shared[gid + 8][byte_col >> 2] >> shift;
            const unsigned h2  = high_shared[gid][(byte_col + 4) >> 2] >> shift;
            const unsigned h3  = high_shared[gid + 8][(byte_col + 4) >> 2] >> shift;
            const unsigned af0 = q5_ksplit_bf16_pair(code_shared[gid][offset], h0);
            const unsigned af1 = q5_ksplit_bf16_pair(code_shared[gid + 8][offset], h1);
            const unsigned af2 = q5_ksplit_bf16_pair(code_shared[gid][offset + 4], h2);
            const unsigned af3 = q5_ksplit_bf16_pair(code_shared[gid + 8][offset + 4], h3);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                         af0, af1, af2, af3, bf[nt][0], bf[nt][1]);
            }
        }

        const float top_scale = __half2float(__ushort_as_half(scale_shared[gid][k_split]));
        const float bot_scale = __half2float(__ushort_as_half(scale_shared[gid + 8][k_split]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
            acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
            acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
            acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
        }
    };

#pragma unroll
    for (int stage = 0; stage + 1 < Stages; ++stage) {
        if (stage < kGroups) {
            stage_weight(stage, stage * kGroupK);
            stage_x(stage, stage * kGroupK);
        }
        cp_commit();
    }
#pragma unroll 1
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        if constexpr (Stages == 1) {
            if (group_index > 0) { __syncthreads(); }
            stage_weight(0, group_index * kGroupK);
            stage_x(0, group_index * kGroupK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
            multiply_group(0);
        } else {
            cp_wait<Stages - 2>();
            __syncthreads();
            const int refill = group_index + Stages - 1;
            if (refill < kGroups) {
                stage_weight(refill % Stages, refill * kGroupK);
                stage_x(refill % Stages, refill * kGroupK);
            }
            cp_commit();
            multiply_group(group_index % Stages);
        }
    }
    cp_wait<0>();
    __syncthreads();

    auto* partial      = shared.partial;
    const auto slot_of = [&](int split, int nt) {
        return partial + ((split * kNt + nt) * 32 + lane) * 4;
    };
    if (k_split != 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(slot_of(k_split, nt),
                      make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();
    if (k_split == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            float4 sum = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
#pragma unroll
            for (int split = 1; split < kWarps; ++split) {
                const float4 value = load_vec<float4>(slot_of(split, nt));
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            epilogue(row0 + gid, nt * 8 + 2 * lid, sum, live);
        }
    }
}

} // namespace ninfer::ops::detail
