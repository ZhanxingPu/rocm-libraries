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
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenRopeGqaForward(miopenHandle_t handle,
                                                int batch,
                                                int seqlen_q,
                                                int seqlen_k,
                                                int nhead_q,
                                                int nhead_k,
                                                int hdim,
                                                int rotary_dim,
                                                void* Q,
                                                void* K,
                                                const void* V,
                                                void* O,
                                                const void* cos_t,
                                                const void* sin_t)
{
    MIOPEN_LOG_FUNCTION(handle,
                        batch,
                        seqlen_q,
                        seqlen_k,
                        nhead_q,
                        nhead_k,
                        hdim,
                        rotary_dim);

    return miopen::try_([&] {
        miopen::RopeGqaForward(miopen::deref(handle),
                               batch,
                               seqlen_q,
                               seqlen_k,
                               nhead_q,
                               nhead_k,
                               hdim,
                               rotary_dim,
                               DataCast(Q),
                               DataCast(K),
                               DataCast(V),
                               DataCast(O),
                               DataCast(cos_t),
                               DataCast(sin_t));
    });
}
