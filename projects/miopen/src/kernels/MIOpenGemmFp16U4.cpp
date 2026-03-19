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
 * RDNA3 WMMA GEMM + Fused Dequantization Kernel
 *
 * Computes C = A × dequant(B_packed)^T  where:
 *   A        : FP16 col-major (M × K), stride lda
 *   B_packed : uint4 packed into bytes (N × K/2), row-major per column
 *   scales   : FP16 (N × num_groups_k), per-column per-group scale
 *   zeros    : FP16 (N × num_groups_k), per-column per-group zero point
 *   C        : FP16 col-major (M × N), stride ldc
 *
 * Requires RDNA3+ (gfx11xx/gfx12xx) for WMMA intrinsics.
 * M must be multiple of 128, N must be multiple of 128, K must be multiple of 32.
 *
 * NO #include allowed — this file is JIT-compiled by MIOpen (comgr/HIPRTC).
 */

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));

#define GEMM_FP16_U4_WMMA_TILE 16
#define GEMM_FP16_U4_BM_R 128
#define GEMM_FP16_U4_BN_R 128
#define GEMM_FP16_U4_BK_W 32
#define GEMM_FP16_U4_WT_MR 2
#define GEMM_FP16_U4_WT_NR 2
#define GEMM_FP16_U4_WM_R (GEMM_FP16_U4_BM_R / (GEMM_FP16_U4_WT_MR * GEMM_FP16_U4_WMMA_TILE))
#define GEMM_FP16_U4_WN_R (GEMM_FP16_U4_BN_R / (GEMM_FP16_U4_WT_NR * GEMM_FP16_U4_WMMA_TILE))
#define GEMM_FP16_U4_NW_R (GEMM_FP16_U4_WM_R * GEMM_FP16_U4_WN_R)
#define GEMM_FP16_U4_THR_R (GEMM_FP16_U4_NW_R * 32)

#ifndef GEMM_FP16_U4_USE_ZEROS
#define GEMM_FP16_U4_USE_ZEROS 0
#endif

extern "C" __global__
    __attribute__((amdgpu_flat_work_group_size(GEMM_FP16_U4_THR_R, GEMM_FP16_U4_THR_R)))
    __attribute__((amdgpu_waves_per_eu(8)))
    void GemmFp16U4FusedWmmaForward(int M,
                                    int N,
                                    int K,
                                    const _Float16* __restrict__ A,
                                    int lda,
                                    const unsigned char* __restrict__ B_packed,
                                    const _Float16* __restrict__ scales,
                                    const _Float16* __restrict__ zeros,
                                    int group_size,
                                    int num_groups_k,
                                    _Float16* __restrict__ C,
                                    int ldc)
{
    constexpr int PAD_K   = 2;
    constexpr int K_STR   = GEMM_FP16_U4_BK_W + PAD_K;
    constexpr int A_GRP   = GEMM_FP16_U4_BM_R / 4;
    constexpr int A_VL    = (GEMM_FP16_U4_BM_R * GEMM_FP16_U4_BK_W) / (GEMM_FP16_U4_THR_R * 4);
    constexpr int K_STEPS = GEMM_FP16_U4_BK_W / GEMM_FP16_U4_WMMA_TILE;

    __shared__ _Float16 smA[2][GEMM_FP16_U4_BM_R][K_STR];
    __shared__ _Float16 smB[2][GEMM_FP16_U4_BN_R][K_STR];

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lid  = tid % 32;
    const int lane = lid % 16;
    const int sub  = lid / 16;
    const int wrow = wid / GEMM_FP16_U4_WN_R;
    const int wcol = wid % GEMM_FP16_U4_WN_R;

    constexpr int SWIZZLE_N = 2;
    const int n_tiles  = gridDim.x;
    const int m_tiles  = gridDim.y;
    const int block_id = blockIdx.y * n_tiles + blockIdx.x;
    const int sw       = (n_tiles >= SWIZZLE_N) ? SWIZZLE_N : n_tiles;
    const int super    = block_id / (sw * m_tiles);
    const int rem      = block_id % (sw * m_tiles);
    int by             = rem / sw;
    int bx             = super * sw + rem % sw;
    if(bx >= n_tiles)
    {
        bx = block_id % n_tiles;
        by = block_id / n_tiles;
    }
    const int row0 = by * GEMM_FP16_U4_BM_R;
    const int col0 = bx * GEMM_FP16_U4_BN_R;

    const int b_col       = tid >> 2;
    const int b_sub       = tid & 3;
    const int b_k8        = b_sub << 3;
    const int b_n_g       = col0 + b_col;
    const int b_byte_base = b_n_g * (K >> 1) + (b_k8 >> 1);
    const int b_gid_base  = b_n_g * num_groups_k;

    int cached_k_grp   = -1;
    _Float16 cached_s  = 0, cached_z = 0;

    float8 acc[GEMM_FP16_U4_WT_MR][GEMM_FP16_U4_WT_NR];
#pragma unroll
    for(int i = 0; i < GEMM_FP16_U4_WT_MR; i++)
#pragma unroll
        for(int j = 0; j < GEMM_FP16_U4_WT_NR; j++)
#pragma unroll
            for(int e = 0; e < 8; e++)
                acc[i][j][e] = 0.0f;

    auto loadTile = [&](int buf, int t) __attribute__((always_inline))
    {
        uint2 a_reg[A_VL];
        int a_m4[A_VL], a_k[A_VL];
#pragma unroll
        for(int ld = 0; ld < A_VL; ld++)
        {
            int vid  = tid + ld * GEMM_FP16_U4_THR_R;
            a_m4[ld] = (vid % A_GRP) * 4;
            a_k[ld]  = vid / A_GRP;
            a_reg[ld] =
                *reinterpret_cast<const uint2*>(&A[(t + a_k[ld]) * lda + row0 + a_m4[ld]]);
        }

        int cur_k_grp = t / group_size;
        if(cur_k_grp != cached_k_grp)
        {
            int gid      = cur_k_grp + b_gid_base;
            cached_s     = scales[gid];
#if GEMM_FP16_U4_USE_ZEROS
            cached_z     = zeros[gid];
#endif
            cached_k_grp = cur_k_grp;
        }
        _Float16 s16 = cached_s, z16 = cached_z;

        unsigned int four =
            *reinterpret_cast<const unsigned int*>(&B_packed[b_byte_base + (t >> 1)]);
        _Float16 bv[8];
#pragma unroll
        for(int i = 0; i < 4; i++)
        {
            unsigned int byte_val = (four >> (i * 8)) & 0xFF;
            bv[i * 2]            = ((_Float16)(int)(byte_val & 0xF) - z16) * s16;
            bv[i * 2 + 1]        = ((_Float16)(int)(byte_val >> 4) - z16) * s16;
        }
        *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = *reinterpret_cast<uint2*>(&bv[0]);
        *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = *reinterpret_cast<uint2*>(&bv[4]);

#pragma unroll
        for(int ld = 0; ld < A_VL; ld++)
        {
            const _Float16* h     = reinterpret_cast<const _Float16*>(&a_reg[ld]);
            smA[buf][a_m4[ld]][a_k[ld]]     = h[0];
            smA[buf][a_m4[ld] + 1][a_k[ld]] = h[1];
            smA[buf][a_m4[ld] + 2][a_k[ld]] = h[2];
            smA[buf][a_m4[ld] + 3][a_k[ld]] = h[3];
        }
    };

    auto computeWMMA = [&](int buf) __attribute__((always_inline))
    {
#pragma unroll
        for(int ks = 0; ks < K_STEPS; ks++)
        {
            half16 b_frag[GEMM_FP16_U4_WT_NR];
#pragma unroll
            for(int wn = 0; wn < GEMM_FP16_U4_WT_NR; wn++)
            {
                int noff              = (wcol * GEMM_FP16_U4_WT_NR + wn) * GEMM_FP16_U4_WMMA_TILE;
                unsigned int* dst     = reinterpret_cast<unsigned int*>(&b_frag[wn]);
                const unsigned int* src = reinterpret_cast<const unsigned int*>(
                    &smB[buf][noff + lane][ks * GEMM_FP16_U4_WMMA_TILE]);
#pragma unroll
                for(int i = 0; i < 8; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for(int wm = 0; wm < GEMM_FP16_U4_WT_MR; wm++)
            {
                int moff              = (wrow * GEMM_FP16_U4_WT_MR + wm) * GEMM_FP16_U4_WMMA_TILE;
                half16 a_frag;
                unsigned int* dst     = reinterpret_cast<unsigned int*>(&a_frag);
                const unsigned int* src = reinterpret_cast<const unsigned int*>(
                    &smA[buf][moff + lane][ks * GEMM_FP16_U4_WMMA_TILE]);
#pragma unroll
                for(int i = 0; i < 8; i++)
                    dst[i] = src[i];

#pragma unroll
                for(int wn = 0; wn < GEMM_FP16_U4_WT_NR; wn++)
                    acc[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        a_frag, b_frag[wn], acc[wm][wn]);
            }
        }
    };

    loadTile(0, 0);
    __syncthreads();

    int buf = 0;
    for(int t = GEMM_FP16_U4_BK_W; t < K; t += GEMM_FP16_U4_BK_W)
    {
        int nxt = 1 - buf;
        loadTile(nxt, t);
        computeWMMA(buf);
        __syncthreads();
        buf = nxt;
    }
    computeWMMA(buf);

#pragma unroll
    for(int wm = 0; wm < GEMM_FP16_U4_WT_MR; wm++)
    {
        int mbase = row0 + (wrow * GEMM_FP16_U4_WT_MR + wm) * GEMM_FP16_U4_WMMA_TILE;
#pragma unroll
        for(int wn = 0; wn < GEMM_FP16_U4_WT_NR; wn++)
        {
            int nbase = col0 + (wcol * GEMM_FP16_U4_WT_NR + wn) * GEMM_FP16_U4_WMMA_TILE;
#pragma unroll
            for(int e = 0; e < 8; e++)
            {
                int r = e * 2 + sub;
                __builtin_nontemporal_store((_Float16)acc[wm][wn][e],
                                            &C[(nbase + lane) * ldc + mbase + r]);
            }
        }
    }
}
