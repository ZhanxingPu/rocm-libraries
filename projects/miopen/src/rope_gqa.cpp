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

#include <miopen/rope_gqa.hpp>
#include <miopen/rope_gqa/invoke_params.hpp>
#include <miopen/rope_gqa/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t RopeGqaForward(const Handle& handle,
                              int batch,
                              int seqlen_q,
                              int seqlen_k,
                              int nhead_q,
                              int nhead_k,
                              int hdim,
                              int rotary_dim,
                              Data_t Q,
                              Data_t K,
                              ConstData_t V,
                              Data_t O,
                              ConstData_t cos_t,
                              ConstData_t sin_t)
{
    const auto problem =
        rope_gqa::ProblemDescription{batch, seqlen_q, seqlen_k,
                                     nhead_q, nhead_k, hdim, rotary_dim};
    const auto invoke_params =
        rope_gqa::RopeGqaInvokeParams{batch, seqlen_q, seqlen_k,
                                       nhead_q, nhead_k, hdim, rotary_dim,
                                       Q, K, V, O, cos_t, sin_t};
    const auto algo    = AlgorithmName{"RopeGqaForward"};
    const auto solvers = solver::SolverContainer<solver::rope_gqa::RopeGqaForward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
