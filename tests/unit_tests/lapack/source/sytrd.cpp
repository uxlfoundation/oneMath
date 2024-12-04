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
1 33 35 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, oneapi::math::uplo uplo, int64_t n, int64_t lda,
              uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda * n);
    std::vector<fp_real> d(n);
    std::vector<fp_real> e(n);
    std::vector<fp> tau(n);
    rand_matrix(seed, oneapi::math::transpose::nontrans, n, n, A, lda);
    std::vector<fp> A_initial = A;

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto d_dev = device_alloc<data_T, fp_real>(queue, d.size());
        auto e_dev = device_alloc<data_T, fp_real>(queue, e.size());
        auto tau_dev = device_alloc<data_T>(queue, tau.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::math::lapack::sytrd_scratchpad_size<fp>(queue, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::math::lapack::sytrd_scratchpad_size<fp>, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, d.data(), d_dev, d.size());
        host_to_device_copy(queue, e.data(), e_dev, e.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::math::lapack::sytrd(queue, uplo, n, A_dev, lda, d_dev, e_dev, tau_dev,
                                    scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::math::lapack::sytrd, uplo, n, A_dev, lda, d_dev,
                                  e_dev, tau_dev, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, d_dev, d.data(), d.size());
        device_to_host_copy(queue, e_dev, e.data(), e.size());
        device_to_host_copy(queue, tau_dev, tau.data(), tau.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, d_dev);
        device_free(queue, e_dev);
        device_free(queue, tau_dev);
        device_free(queue, scratchpad_dev);
    }

    hermitian_to_full(uplo, n, A_initial, lda);
    bool result = true;

    /* |A - Q T Q'| < |A| O(eps) */
    std::vector<fp> T(n * n);
    int64_t ldt = n;
    for (int64_t diag = 0; diag < n; diag++)
        T[diag + diag * ldt] = d[diag];
    for (int64_t diag = 0; diag < n - 1; diag++) {
        T[diag + (diag + 1) * ldt] = e[diag];
        T[diag + 1 + (diag)*ldt] = e[diag];
    }

    std::vector<fp> QTQ{ T };
    int64_t ldqtq = n;
    reference::or_un_mtr(oneapi::math::side::left, uplo, oneapi::math::transpose::nontrans, n, n,
                         A.data(), lda, tau.data(), QTQ.data(), ldqtq);
    reference::or_un_mtr(oneapi::math::side::right, uplo, oneapi::math::transpose::trans, n, n,
                         A.data(), lda, tau.data(), QTQ.data(), ldqtq);

    if (!rel_mat_err_check(n, n, QTQ, ldqtq, A_initial, lda)) {
        test_log::lout << "Factorization check failed" << std::endl;
        result = false;
    }

    /* A[i, i] = d[i] */
    for (int64_t diag = 0; diag < n; diag++)
        d[diag] -= A[diag + diag * lda];
    if (uplo == oneapi::math::uplo::upper)
        for (int64_t diag = 0; diag < n - 1; diag++)
            e[diag] -= A[diag + (diag + 1) * lda];
    else
        for (int64_t diag = 0; diag < n - 1; diag++)
            e[diag] -= A[diag + 1 + (diag)*lda];

    auto ulp = reference::lamch<fp_real>('P');
    if (reference::lange('I', n, 1, d.data(), n) > 10.0 * ulp) {
        test_log::lout << "Diagonal check failed" << std::endl;
        result = false;
    }
    if (reference::lange('I', n - 1, 1, e.data(), n - 1) > 10.0 * ulp) {
        test_log::lout << "Off-diagonal check failed" << std::endl;
        result = false;
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
    std::vector<fp> A(lda * n);
    std::vector<fp_real> d(n);
    std::vector<fp_real> e(n);
    std::vector<fp> tau(n);
    rand_matrix(seed, oneapi::math::transpose::nontrans, n, n, A, lda);
    std::vector<fp> A_initial = A;

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto d_dev = device_alloc<data_T, fp_real>(queue, d.size());
        auto e_dev = device_alloc<data_T, fp_real>(queue, e.size());
        auto tau_dev = device_alloc<data_T>(queue, tau.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::math::lapack::sytrd_scratchpad_size<fp>(queue, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::math::lapack::sytrd_scratchpad_size<fp>, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, d.data(), d_dev, d.size());
        host_to_device_copy(queue, e.data(), e_dev, e.size());
        host_to_device_copy(queue, tau.data(), tau_dev, tau.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::math::lapack::sytrd(
            queue, uplo, n, A_dev, lda, d_dev, e_dev, tau_dev, scratchpad_dev, scratchpad_size,
            std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::math::lapack::sytrd, uplo, n, A_dev,
                                  lda, d_dev, e_dev, tau_dev, scratchpad_dev, scratchpad_size,
                                  std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, d_dev);
        device_free(queue, e_dev);
        device_free(queue, tau_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY_REAL(Sytrd);
INSTANTIATE_GTEST_SUITE_DEPENDENCY_REAL(Sytrd);
