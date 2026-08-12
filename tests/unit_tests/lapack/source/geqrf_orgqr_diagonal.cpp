/*******************************************************************************
* Copyright 2025 Intel Corporation
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

/*
 * Regression test for https://github.com/uxlfoundation/oneMath/issues/626.
 *
 * QR decomposition (geqrf + orgqr) on the cuSOLVER backend used to return
 * incorrect results on repeated runs sharing a queue: for a diagonal input
 * matrix the diagonal of Q should be 1, but from the second run onwards some
 * entries stayed at the input value because the native command completed
 * before the cuSOLVER kernels had actually finished. This test performs the
 * geqrf + orgqr sequence several times on a known diagonal matrix and verifies
 * that every run yields a Q whose diagonal is 1 (and off-diagonal 0), guarding
 * against a regression of that synchronization bug.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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

/* columns: n, lda, num_runs */
const char* accuracy_input = R"(
256 256 6
512 512 6
1024 1024 6
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, int64_t n, int64_t lda, int64_t num_runs) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    const int64_t m = n;
    const int64_t k = n;

    /* Diagonal input matrix (column-major) with a non-unit diagonal value.
     * Its QR factor Q is the identity, so Q's diagonal must be 1. */
    const fp diag_value = fp(2.0);
    std::vector<fp> A_initial(lda * n, fp(0.0));
    for (int64_t j = 0; j < n; ++j) {
        A_initial[j * lda + j] = diag_value;
    }

    const fp_real tol = fp_real(10.0) * n * std::numeric_limits<fp_real>::epsilon();

    bool result = true;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A_initial.size());
        auto tau_dev = device_alloc<data_T>(queue, k);
#ifdef CALL_RT_API
        const auto geqrf_scratchpad_size =
            oneapi::math::lapack::geqrf_scratchpad_size<fp>(queue, m, n, lda);
        const auto orgqr_scratchpad_size =
            oneapi::math::lapack::orgqr_scratchpad_size<fp>(queue, m, n, k, lda);
#else
        int64_t geqrf_scratchpad_size;
        int64_t orgqr_scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, geqrf_scratchpad_size = oneapi::math::lapack::geqrf_scratchpad_size<fp>, m, n,
            lda);
        TEST_RUN_LAPACK_CT_SELECT(
            queue, orgqr_scratchpad_size = oneapi::math::lapack::orgqr_scratchpad_size<fp>, m, n, k,
            lda);
#endif
        const auto scratchpad_size = std::max(geqrf_scratchpad_size, orgqr_scratchpad_size);
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        std::vector<fp> Q(lda * n);
        for (int64_t run = 0; run < num_runs && result; ++run) {
            /* Reset the input for every run and factor it in place. */
            host_to_device_copy(queue, A_initial.data(), A_dev, A_initial.size());
            queue.wait_and_throw();

#ifdef CALL_RT_API
            oneapi::math::lapack::geqrf(queue, m, n, A_dev, lda, tau_dev, scratchpad_dev,
                                        geqrf_scratchpad_size);
            oneapi::math::lapack::orgqr(queue, m, n, k, A_dev, lda, tau_dev, scratchpad_dev,
                                        orgqr_scratchpad_size);
#else
            TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::math::lapack::geqrf, m, n, A_dev, lda, tau_dev,
                                      scratchpad_dev, geqrf_scratchpad_size);
            TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::math::lapack::orgqr, m, n, k, A_dev, lda,
                                      tau_dev, scratchpad_dev, orgqr_scratchpad_size);
#endif
            queue.wait_and_throw();

            device_to_host_copy(queue, A_dev, Q.data(), Q.size());
            queue.wait_and_throw();

            for (int64_t j = 0; j < n && result; ++j) {
                for (int64_t i = 0; i < m && result; ++i) {
                    const fp_real expected = (i == j) ? fp_real(1.0) : fp_real(0.0);
                    const fp_real got = std::abs(Q[j * lda + i]);
                    if (std::abs(got - expected) > tol) {
                        test_log::lout << "run " << run << ": Q(" << i << ", " << j << ") = " << got
                                       << ", expected " << expected << std::endl;
                        result = false;
                    }
                }
            }
        }

        device_free(queue, A_dev);
        device_free(queue, tau_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY_REAL(GeqrfOrgqrDiagonal);
