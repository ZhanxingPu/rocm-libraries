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
#include <miopen/gemm_fp16_u4.hpp>
#include <miopen/gemm_fp16_u4/invoke_params.hpp>
#include <miopen/gemm_fp16_u4/solvers.hpp>
#include <miopen/find_solution.hpp>

namespace miopen {

miopenStatus_t GemmFp16U4Forward(const Handle& handle,
                                 int M,
                                 int N,
                                 int K,
                                 ConstData_t A,
                                 int lda,
                                 ConstData_t B_packed,
                                 ConstData_t scales,
                                 ConstData_t zeros,
                                 int group_size,
                                 int num_groups_k,
                                 Data_t C,
                                 int ldc)
{
    const auto problem =
        gemm_fp16_u4::ProblemDescription{M, N, K, group_size, num_groups_k,
                                         zeros != nullptr};
    const auto invoke_params =
        gemm_fp16_u4::GemmFp16U4InvokeParams{M, N, K, A, lda, B_packed, scales, zeros,
                                              group_size, num_groups_k, C, ldc};
    const auto algo    = AlgorithmName{"GemmFp16U4Forward"};
    const auto solvers = solver::SolverContainer<solver::gemm_fp16_u4::GemmFp16U4Forward>{};
    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
