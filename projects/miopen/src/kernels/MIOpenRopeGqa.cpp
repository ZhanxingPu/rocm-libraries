/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/*
 * ============================================================
 * MIOpenRopeGqa — RoPE + GQA Flash Attention Kernel
 * ============================================================
 *
 * 本文件包含 rope_gqa 操作的全部 GPU 计算逻辑，编译期链接进 MIOpen.dll:
 *
 *   Part 1: RoPE (Rotary Position Embedding)
 *           HIP __global__ kernel, LLaMA 风格 half-rotation.
 *             x'[d]      = x[d]*cos - x[d+half]*sin
 *             x'[d+half] = x[d]*sin + x[d+half]*cos
 *
 *   Part 2: FMHA GQA (Flash Multi-Head Attention)
 *           CK Tile FmhaFwdKernel 模板实例化.
 *             O = softmax(Q'K'^T / sqrt(hdim)) × V
 *           FP16 I/O, FP32 accumulation, GQA ratio = nhead_q / nhead_k.
 *
 * 入口函数: RunRopeGqaKernel()
 *   RoPE(Q) → RoPE(K) → FMHA GQA → O
 *
 * 编译条件: -DMIOPEN_USE_CK_TILE_FMHA=1 -DCK_TILE_USE_WMMA=1
 * ============================================================
 */

#ifdef MIOPEN_USE_CK_TILE_FMHA

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>

// ============================================================
// Part 1: RoPE GPU Kernel
// ============================================================
__global__ void RopeGqaKernel(
    __half* __restrict__ x,
    const __half* __restrict__ cos_t,
    const __half* __restrict__ sin_t,
    int total_rows,
    int seqlen,
    int hdim,
    int half_rot)
{
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(d >= half_rot || row >= total_rows)
        return;

    int s = row % seqlen;
    size_t base = static_cast<size_t>(row) * hdim;

    float x1 = __half2float(x[base + d]);
    float x2 = __half2float(x[base + d + half_rot]);
    float c  = __half2float(cos_t[s * half_rot + d]);
    float sn = __half2float(sin_t[s * half_rot + d]);

    x[base + d]            = __float2half(x1 * c - x2 * sn);
    x[base + d + half_rot] = __float2half(x1 * sn + x2 * c);
}

// ============================================================
// Part 2: CK Tile FMHA GQA
// ============================================================
#define CK_TILE_FMHA_FWD_FAST_EXP2 0

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

namespace {

using QDataType   = ck_tile::half_t;
using KDataType   = ck_tile::half_t;
using VDataType   = ck_tile::half_t;
using BiasDataType = ck_tile::half_t;
using RandValOutputDataType = uint8_t;
using LSEDataType  = float;
using SaccDataType = float;
using SMPLComputeDataType = float;
using PDataType    = ck_tile::half_t;
using OaccDataType = float;
using ODataType    = ck_tile::half_t;

using fmha_block_tile = ck_tile::sequence<128, 64, 32, 128, 32, 128>;

using fmha_shape = ck_tile::TileFmhaShape<
    fmha_block_tile,
    ck_tile::sequence<8, 1, 1>,
    ck_tile::sequence<16, 16, 16>,
    ck_tile::sequence<8, 1, 1>,
    ck_tile::sequence<16, 16, 16>,
    true>;

using fmha_traits = ck_tile::TileFmhaTraits<
    true, true, true, true,
    false,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    false, false, false,
    ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
    6>;

using fmha_variant = ck_tile::ComposedAttention<0, false>;
using fmha_mask = ck_tile::GenericAttentionMask<false>;

using fmha_pipeline_problem = ck_tile::BlockFmhaPipelineProblem<
    QDataType, KDataType, VDataType,
    SaccDataType, SMPLComputeDataType,
    BiasDataType, RandValOutputDataType, LSEDataType,
    PDataType, OaccDataType, ODataType,
    fmha_shape, false, fmha_variant, fmha_mask, false, fmha_traits>;

using fmha_pipeline = ck_tile::BlockFmhaPipelineQRKSVS<fmha_pipeline_problem>;

using fmha_epilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<OaccDataType, ODataType, true, true>>;

using fmha_kernel_t = ck_tile::FmhaFwdKernel<fmha_pipeline, fmha_epilogue>;

constexpr int ROPE_TARGET_THREADS = 512;

} // anonymous namespace

// ============================================================
// Entry: RoPE(Q) + RoPE(K) + FMHA GQA
// ============================================================
void RunRopeGqaKernel(
    hipStream_t stream,
    int batch, int seqlen_q, int seqlen_k,
    int nhead_q, int nhead_k, int hdim, int rotary_dim,
    void* Q, void* K, const void* V, void* O,
    const void* cos_t, const void* sin_t)
{
    int half_rot = rotary_dim / 2;

    // RoPE on Q
    {
        int total_rows = batch * nhead_q * seqlen_q;
        int bx = std::min(half_rot, ROPE_TARGET_THREADS);
        int by = std::max(1, ROPE_TARGET_THREADS / bx);
        hipLaunchKernelGGL(RopeGqaKernel,
                           dim3((half_rot + bx - 1) / bx, (total_rows + by - 1) / by),
                           dim3(bx, by), 0, stream,
                           static_cast<__half*>(Q),
                           static_cast<const __half*>(cos_t),
                           static_cast<const __half*>(sin_t),
                           total_rows, seqlen_q, hdim, half_rot);
    }

    // RoPE on K
    {
        int total_rows = batch * nhead_k * seqlen_k;
        int bx = std::min(half_rot, ROPE_TARGET_THREADS);
        int by = std::max(1, ROPE_TARGET_THREADS / bx);
        hipLaunchKernelGGL(RopeGqaKernel,
                           dim3((half_rot + bx - 1) / bx, (total_rows + by - 1) / by),
                           dim3(bx, by), 0, stream,
                           static_cast<__half*>(K),
                           static_cast<const __half*>(cos_t),
                           static_cast<const __half*>(sin_t),
                           total_rows, seqlen_k, hdim, half_rot);
    }

    // FMHA GQA
    {
        int ratio = nhead_q / nhead_k;
        float scale_s = 1.0f / std::sqrt(static_cast<float>(hdim));

        auto kargs = fmha_kernel_t::MakeKargs(
            Q, K, V,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, O,
            seqlen_q, seqlen_k, hdim, hdim,
            nhead_q, ratio, scale_s, 0.0f,
            hdim, hdim, hdim, 0, 0, hdim,
            seqlen_q * hdim, seqlen_k * hdim, seqlen_k * hdim,
            0, 0, 0, seqlen_q * hdim,
            0, 0, 0,
            nhead_q * seqlen_q * hdim,
            nhead_k * seqlen_k * hdim,
            nhead_k * seqlen_k * hdim,
            0, 0, 0,
            nhead_q * seqlen_q * hdim,
            0, 0, 0,
            0, 0, 0, 0,
            0.0f, false,
            std::make_tuple(uint64_t(0), uint64_t(0)), 0, 0);

        dim3 grids  = fmha_kernel_t::GridSize(batch, nhead_q, seqlen_q, hdim, false);
        dim3 blocks = fmha_kernel_t::BlockSize();

        auto kern = ck_tile::make_kernel<6>(fmha_kernel_t{}, grids, blocks, 0, kargs);
        ck_tile::stream_config sc{stream, false, 0, 0, 0, false};
        kern(sc);
    }
}

#endif // MIOPEN_USE_CK_TILE_FMHA
