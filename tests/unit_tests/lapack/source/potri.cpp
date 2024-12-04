/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#include <complex>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math.hpp"
#include "lapack_common.hpp"
#include "lapack_test_controller.hpp"
#include "lapack_accuracy_checks.hpp"
#include "lapack_reference_wrappers.hpp"
#include "test_helper.hpp"

namespace {

const char* accuracy_input = R"(
0 30 42 27182
0 33 33 27182
1 31 41 27182
1 45 45 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, oneapi::math::uplo uplo, int64_t n, int64_t lda,
              uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(n * lda);
    rand_pos_def_matrix(seed, uplo, n, A_initial, lda);
    std::vector<fp> A = A_initial;

    auto info = reference::potrf(uplo, n, A.data(), lda);
    if (0 != info) {
        test_log::lout << "Reference potrf failed with info: " << info << std::endl;
        return false;
    }

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };
        auto A_dev = device_alloc<data_T>(queue, A.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::math::lapack::potri_scratchpad_size<fp>(queue, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::math::lapack::potri_scratchpad_size<fp>, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::math::lapack::potri(queue, uplo, n, A_dev, lda, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::math::lapack::potri, uplo, n, A_dev, lda,
                                  scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, scratchpad_dev);
    }

    hermitian_to_full(uplo, n, A, lda);
    hermitian_to_full(uplo, n, A_initial, lda);

    auto norm_A = reference::lange('1', n, n, A_initial.data(), lda);
    auto norm_Ainv = reference::lange('1', n, n, A.data(), lda);
    auto ulp = reference::lamch<fp_real>('P');

    std::vector<fp> resid(n * n);
    int64_t ldr = n;
    for (int64_t diag = 0; diag < n; diag++)
        resid[diag + diag * ldr] = static_cast<fp>(1.0);
    reference::gemm(oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, n, n, n,
                    1.0, A_initial.data(), lda, A.data(), lda, -1.0, resid.data(), ldr);
    auto norm_resid = reference::lange('1', n, n, resid.data(), ldr);
    auto rel_err = norm_resid / (norm_A * norm_Ainv * n * ulp);
    fp_real threshold = 30.0;

    bool result = rel_err < threshold;
    if (!result) {
        snprintf(test_log::buffer.data(), test_log::buffer.size(),
                 "|A inv(A) - I| / ( |A| |inv(A)| n ulp ) = |%e|/(|%e|*|%e|*%d*%e) = %e\n",
                 norm_resid, norm_A, norm_Ainv, static_cast<int>(n), ulp, rel_err);
        test_log::lout << test_log::buffer.data();
        test_log::lout << "threshold = " << threshold << std::endl;
    }

    return result;
}

const char* dependency_input = R"(
1 1 1 1
)";

template <typename data_T>
bool usm_dependency(const sycl::device& dev, oneapi::math::uplo uplo, int64_t n, int64_t lda,
                    uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(n * lda);
    rand_pos_def_matrix(seed, uplo, n, A_initial, lda);
    std::vector<fp> A = A_initial;

    auto info = reference::potrf(uplo, n, A.data(), lda);
    if (0 != info) {
        test_log::lout << "Reference potrf failed with info: " << info << std::endl;
        return false;
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };
        auto A_dev = device_alloc<data_T>(queue, A.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::math::lapack::potri_scratchpad_size<fp>(queue, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::math::lapack::potri_scratchpad_size<fp>, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event =
            oneapi::math::lapack::potri(queue, uplo, n, A_dev, lda, scratchpad_dev, scratchpad_size,
                                        std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::math::lapack::potri, uplo, n, A_dev,
                                  lda, scratchpad_dev, scratchpad_size,
                                  std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY(Potri);
INSTANTIATE_GTEST_SUITE_DEPENDENCY(Potri);
