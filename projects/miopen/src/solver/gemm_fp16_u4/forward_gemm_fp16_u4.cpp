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

#include <miopen/gemm_fp16_u4/solvers.hpp>

#include <miopen/gemm_fp16_u4/invoke_params.hpp>
#include <miopen/gemm_fp16_u4.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

namespace miopen {

namespace solver {

namespace gemm_fp16_u4 {

bool GemmFp16U4Forward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                     const miopen::gemm_fp16_u4::ProblemDescription& problem) const
{
    int M = problem.GetM();
    int N = problem.GetN();
    int K = problem.GetK();

    if(M % 128 != 0 || N % 128 != 0 || K % 32 != 0)
        return false;

    return true;
}

ConvSolution GemmFp16U4Forward::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                            const miopen::gemm_fp16_u4::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    int M = problem.GetM();
    int N = problem.GetN();

    constexpr int BM_R  = 128;
    constexpr int BN_R  = 128;
    constexpr int THR_R = 512;

    size_t grid_x = static_cast<size_t>(N / BN_R) * THR_R;
    size_t grid_y = static_cast<size_t>(M / BM_R);

    KernelBuildParameters build_params;

    if(problem.HasZeros())
        build_params.Define("GEMM_FP16_U4_USE_ZEROS", 1);

    auto kernel = KernelInfo{};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(THR_R);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    kernel.g_wk.push_back(grid_x);
    kernel.g_wk.push_back(grid_y);
    kernel.g_wk.push_back(1);

    kernel.kernel_file = "MIOpenGemmFp16U4.cpp";
    kernel.kernel_name = "GemmFp16U4FusedWmmaForward";

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) k = handle_.Run(kernels.front());
            decltype(auto) params =
                raw_params.CastTo<miopen::gemm_fp16_u4::GemmFp16U4InvokeParams>();

            k(params.M,
              params.N,
              params.K,
              params.A,
              params.lda,
              params.B_packed,
              params.scales,
              params.zeros,
              params.group_size,
              params.num_groups_k,
              params.C,
              params.ldc);
        };
    };

    result.construction_params.push_back(kernel);

    return result;
}

} // namespace gemm_fp16_u4

} // namespace solver

} // namespace miopen
