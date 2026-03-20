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
namespace rope_gqa {

struct RopeGqaInvokeParams : public miopen::InvokeParams
{
    RopeGqaInvokeParams(int batch_,
                        int seqlen_q_,
                        int seqlen_k_,
                        int nhead_q_,
                        int nhead_k_,
                        int hdim_,
                        int rotary_dim_,
                        Data_t Q_,
                        Data_t K_,
                        ConstData_t V_,
                        Data_t O_,
                        ConstData_t cos_t_,
                        ConstData_t sin_t_)
        : batch(batch_),
          seqlen_q(seqlen_q_),
          seqlen_k(seqlen_k_),
          nhead_q(nhead_q_),
          nhead_k(nhead_k_),
          hdim(hdim_),
          rotary_dim(rotary_dim_),
          Q(Q_),
          K(K_),
          V(V_),
          O(O_),
          cos_t(cos_t_),
          sin_t(sin_t_)
    {
    }

    int batch;
    int seqlen_q;
    int seqlen_k;
    int nhead_q;
    int nhead_k;
    int hdim;
    int rotary_dim;
    Data_t Q         = nullptr;
    Data_t K         = nullptr;
    ConstData_t V    = nullptr;
    Data_t O         = nullptr;
    ConstData_t cos_t = nullptr;
    ConstData_t sin_t = nullptr;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace rope_gqa
} // namespace miopen
