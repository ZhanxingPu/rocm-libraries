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

#include <miopen/invoke_params.hpp>

namespace miopen {
namespace gemmdq {

struct GemmDqInvokeParams : public miopen::InvokeParams
{
    GemmDqInvokeParams(int M_,
                       int N_,
                       int K_,
                       ConstData_t A_,
                       int lda_,
                       ConstData_t B_packed_,
                       ConstData_t scales_,
                       ConstData_t zeros_,
                       int group_size_,
                       int num_groups_k_,
                       Data_t C_,
                       int ldc_)
        : M(M_),
          N(N_),
          K(K_),
          A(A_),
          lda(lda_),
          B_packed(B_packed_),
          scales(scales_),
          zeros(zeros_),
          group_size(group_size_),
          num_groups_k(num_groups_k_),
          C(C_),
          ldc(ldc_)
    {
    }

    int M;
    int N;
    int K;
    ConstData_t A         = nullptr;
    int lda;
    ConstData_t B_packed  = nullptr;
    ConstData_t scales    = nullptr;
    ConstData_t zeros     = nullptr;
    int group_size;
    int num_groups_k;
    Data_t C              = nullptr;
    int ldc;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace gemmdq
} // namespace miopen
