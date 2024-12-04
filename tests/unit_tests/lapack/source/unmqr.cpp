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
0 0 50 70 50 70 70 27182
0 0 20 22 20 22 22 27182
0 3 50 70 50 70 70 27182
0 3 20 22 20 22 22 27182
1 0 50 70 70 90 70 27182
1 0 20 22 22 24 22 27182
1 3 50 70 70 90 70 27182
1 3 20 22 22 24 22 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, oneapi::math::side left_right, oneapi::math::transpose trans,
              int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldc, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> C_initial(ldc * n);
    rand_matrix(seed, oneapi::math::transpose::nontrans, m, n, C_initial, ldc);
    std::vector<fp> C = C_initial;

    int64_t nq = (left_right == oneapi::math::side::left) ? m : n;
    std::vector<fp> A(lda * k);
    rand_matrix(seed, oneapi::math::transpose::nontrans, nq, k, A, lda);
    std::vector<fp> tau(k);

    auto info = reference::geqrf(nq, k, A.data(), lda, tau.data());
    if (0 != info) {
        test_log::lout << "reference geqrf failed with info: " << info << std::endl;
        return false;
    }

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };
        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto tau_dev = device_alloc<data_T>(queue, tau.size());
        auto C_dev = device_alloc<data_T>(queue, C.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::math::lapack::unmqr_scratchpad_size<fp>(
            queue, left_right, trans, m, n, k, lda, ldc);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(queue,
                                  scratchpad_size = oneapi::math::lapack::unmqr_scratchpad_size<fp>,
                                  left_right, trans, m, n, k, lda, ldc);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        host_to_device_copy(queue, C.data(), C_dev, C.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::math::lapack::unmqr(queue, left_right, trans, m, n, k, A_dev, lda, tau_dev, C_dev,
                                    ldc, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::math::lapack::unmqr, left_right, trans, m, n, k,
                                  A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, C_dev, C.data(), C.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, C_dev);
        device_free(queue, scratchpad_dev);
    }
    bool result = true;

    /* |Q C - QC| < |QC| O(eps) */
    const auto& QC = C;
    auto& QC_ref = C_initial;
    auto ldqc = ldc;
    info = reference::or_un_mqr(left_right, trans, m, n, k, A.data(), lda, tau.data(),
                                QC_ref.data(), ldqc);
    if (0 != info) {
        test_log::lout << "reference unmqr failed with info: " << info << std::endl;
        return false;
    }
    if (!rel_mat_err_check(m, n, QC, ldqc, QC_ref, ldqc, 1.0)) {
        test_log::lout << "Multiplication check failed" << std::endl;
        result = false;
    }
    return result;
}

const char* dependency_input = R"(
0 0 1 1 1 1 1 1
)";

template <typename data_T>
bool usm_dependency(const sycl::device& dev, oneapi::math::side left_right,
                    oneapi::math::transpose trans, int64_t m, int64_t n, int64_t k, int64_t lda,
                    int64_t ldc, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> C_initial(ldc * n);
    rand_matrix(seed, oneapi::math::transpose::nontrans, m, n, C_initial, ldc);
    std::vector<fp> C = C_initial;

    int64_t nq = (left_right == oneapi::math::side::left) ? m : n;
    std::vector<fp> A(lda * k);
    rand_matrix(seed, oneapi::math::transpose::nontrans, nq, k, A, lda);
    std::vector<fp> tau(k);

    auto info = reference::geqrf(nq, k, A.data(), lda, tau.data());
    if (0 != info) {
        test_log::lout << "reference geqrf failed with info: " << info << std::endl;
        return false;
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto tau_dev = device_alloc<data_T>(queue, tau.size());
        auto C_dev = device_alloc<data_T>(queue, C.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::math::lapack::unmqr_scratchpad_size<fp>(
            queue, left_right, trans, m, n, k, lda, ldc);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(queue,
                                  scratchpad_size = oneapi::math::lapack::unmqr_scratchpad_size<fp>,
                                  left_right, trans, m, n, k, lda, ldc);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        host_to_device_copy(queue, C.data(), C_dev, C.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::math::lapack::unmqr(
            queue, left_right, trans, m, n, k, A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev,
            scratchpad_size, std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::math::lapack::unmqr, left_right,
                                  trans, m, n, k, A_dev, lda, tau_dev, C_dev, ldc, scratchpad_dev,
                                  scratchpad_size, std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, C_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY_COMPLEX(Unmqr);
INSTANTIATE_GTEST_SUITE_DEPENDENCY_COMPLEX(Unmqr);
