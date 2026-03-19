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

#pragma once

#include <miopen/problem_description_base.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace gemm_fp16_u4 {

// Fused GEMM + Dequantization (RDNA3 WMMA)
//
// Fixed data types (enforced by kernel design, not runtime-checkable via raw pointers):
//   A        : FP16  (_Float16)    — input activations, col-major
//   B_packed : UINT4 (packed u8)   — quantized weights, 2 values per byte
//   scales   : FP16  (_Float16)    — per-group scale factors
//   zeros    : FP16  (_Float16)    — per-group zero points
//   C        : FP16  (_Float16)    — output, col-major
//
// Passing data of other types will produce incorrect results silently.
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(int M_, int N_, int K_, int group_size_, int num_groups_k_)
        : M(M_), N(N_), K(K_), group_size(group_size_), num_groups_k(num_groups_k_)
    {
        if(M <= 0 || N <= 0 || K <= 0)
        {
            MIOPEN_THROW(miopenStatusBadParm, "GemmFp16U4Forward: M, N, K must be positive.");
        }
        if(M % 128 != 0 || N % 128 != 0 || K % 32 != 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GemmFp16U4Forward: M must be multiple of 128, "
                         "N must be multiple of 128, K must be multiple of 32.");
        }
        if(group_size <= 0 || num_groups_k <= 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GemmFp16U4Forward: group_size and num_groups_k must be positive.");
        }
        if(K != group_size * num_groups_k)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GemmFp16U4Forward: K must equal group_size * num_groups_k.");
        }
    }

    int GetM() const { return M; }
    int GetN() const { return N; }
    int GetK() const { return K; }
    int GetGroupSize() const { return group_size; }
    int GetNumGroupsK() const { return num_groups_k; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    int M;
    int N;
    int K;
    int group_size;
    int num_groups_k;
};

} // namespace gemm_fp16_u4
} // namespace miopen
