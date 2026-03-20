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

#include <miopen/rope_gqa/solvers.hpp>

#include <miopen/rope_gqa/invoke_params.hpp>
#include <miopen/rope_gqa.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/target_properties.hpp>

#ifdef MIOPEN_USE_CK_TILE_FMHA
extern void RunRopeGqaKernel(
    hipStream_t stream,
    int batch, int seqlen_q, int seqlen_k,
    int nhead_q, int nhead_k, int hdim, int rotary_dim,
    void* Q, void* K, const void* V, void* O,
    const void* cos_t, const void* sin_t);
#endif

namespace miopen {

namespace solver {

namespace rope_gqa {

bool RopeGqaForward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                                  const miopen::rope_gqa::ProblemDescription& problem) const
{
#ifndef MIOPEN_USE_CK_TILE_FMHA
    (void)problem;
    return false;
#else
    int hdim = problem.GetHdim();

    if(hdim != 128)
        return false;

    return true;
#endif
}

ConvSolution RopeGqaForward::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                         const miopen::rope_gqa::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

#ifdef MIOPEN_USE_CK_TILE_FMHA
    std::ignore = problem;

    result.invoker_factory = [](const std::vector<Kernel>&) {
        return [](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params =
                raw_params.CastTo<miopen::rope_gqa::RopeGqaInvokeParams>();

            RunRopeGqaKernel(
                handle_.GetStream(),
                params.batch, params.seqlen_q, params.seqlen_k,
                params.nhead_q, params.nhead_k, params.hdim, params.rotary_dim,
                params.Q, params.K, params.V, params.O,
                params.cos_t, params.sin_t);
        };
    };
#endif

    return result;
}

} // namespace rope_gqa

} // namespace solver

} // namespace miopen
