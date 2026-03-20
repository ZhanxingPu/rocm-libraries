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

namespace rope_gqa {

// RoPE + GQA Flash Attention Forward
//
// Pipeline: RoPE(Q, K) → CK Tile FMHA GQA → O
//
// Tensors (all FP16):
//   Q     : [batch, nhead_q, seqlen_q, hdim]   — in-place (modified by RoPE)
//   K     : [batch, nhead_k, seqlen_k, hdim]   — in-place (modified by RoPE)
//   V     : [batch, nhead_k, seqlen_k, hdim]   — read-only
//   O     : [batch, nhead_q, seqlen_q, hdim]   — output
//   cos_t : [max(seqlen_q, seqlen_k), rotary_dim/2]
//   sin_t : [max(seqlen_q, seqlen_k), rotary_dim/2]
//
// Constraints: nhead_q % nhead_k == 0, rotary_dim even, rotary_dim <= hdim
struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(int batch_,
                       int seqlen_q_,
                       int seqlen_k_,
                       int nhead_q_,
                       int nhead_k_,
                       int hdim_,
                       int rotary_dim_)
        : batch(batch_),
          seqlen_q(seqlen_q_),
          seqlen_k(seqlen_k_),
          nhead_q(nhead_q_),
          nhead_k(nhead_k_),
          hdim(hdim_),
          rotary_dim(rotary_dim_)
    {
        if(batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RopeGqaForward: batch, seqlen_q, seqlen_k must be positive.");
        }
        if(nhead_q <= 0 || nhead_k <= 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RopeGqaForward: nhead_q and nhead_k must be positive.");
        }
        if(nhead_q % nhead_k != 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RopeGqaForward: nhead_q must be divisible by nhead_k.");
        }
        if(hdim <= 0 || rotary_dim <= 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RopeGqaForward: hdim and rotary_dim must be positive.");
        }
        if(rotary_dim % 2 != 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RopeGqaForward: rotary_dim must be even.");
        }
        if(rotary_dim > hdim)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "RopeGqaForward: rotary_dim must be <= hdim.");
        }
    }

    int GetBatch() const { return batch; }
    int GetSeqlenQ() const { return seqlen_q; }
    int GetSeqlenK() const { return seqlen_k; }
    int GetNheadQ() const { return nhead_q; }
    int GetNheadK() const { return nhead_k; }
    int GetHdim() const { return hdim; }
    int GetRotaryDim() const { return rotary_dim; }
    int GetRatio() const { return nhead_q / nhead_k; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    int batch;
    int seqlen_q;
    int seqlen_k;
    int nhead_q;
    int nhead_k;
    int hdim;
    int rotary_dim;
};

} // namespace rope_gqa
} // namespace miopen
