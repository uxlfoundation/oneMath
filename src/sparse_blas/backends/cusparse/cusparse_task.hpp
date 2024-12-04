/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef _ONEMATH_SPARSE_BLAS_BACKENDS_CUSPARSE_TASK_HPP_
#define _ONEMATH_SPARSE_BLAS_BACKENDS_CUSPARSE_TASK_HPP_

#include "cusparse_error.hpp"
#include "sparse_blas/backends/common_launch_task.hpp"

namespace oneapi::math::sparse::cusparse::detail {

// Helper function for functors submitted to host_task or native_command.
// When the extension is disabled, host_task are used and the synchronization is needed to ensure the sycl::event corresponds to the end of the whole functor.
// When the extension is enabled, host_task are still used for out-of-order queues, see description of dispatch_submit_impl_fp_int.
inline void synchronize_if_needed(bool is_in_order_queue, CUstream cu_stream) {
#ifndef SYCL_EXT_ONEAPI_ENQUEUE_NATIVE_COMMAND
    (void)is_in_order_queue;
    CUDA_ERROR_FUNC(cuStreamSynchronize, cu_stream);
#else
    if (!is_in_order_queue) {
        CUDA_ERROR_FUNC(cuStreamSynchronize, cu_stream);
    }
#endif
}

} // namespace oneapi::math::sparse::cusparse::detail

#endif // _ONEMATH_SPARSE_BLAS_BACKENDS_CUSPARSE_TASK_HPP_
