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
0 29 36 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, oneapi::math::uplo uplo, int64_t n, int64_t lda,
              uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda * n);
    std::vector<int64_t> ipiv(n);
    rand_symmetric_matrix(seed, uplo, n, A, lda);

    std::vector<fp> A_initial = A;

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto ipiv_dev = device_alloc<data_T, int64_t>(queue, ipiv.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::math::lapack::sytrf_scratchpad_size<fp>(queue, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::math::lapack::sytrf_scratchpad_size<fp>, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, ipiv.data(), ipiv_dev, ipiv.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::math::lapack::sytrf(queue, uplo, n, A_dev, lda, ipiv_dev, scratchpad_dev,
                                    scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::math::lapack::sytrf, uplo, n, A_dev, lda, ipiv_dev,
                                  scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, ipiv_dev, ipiv.data(), ipiv.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, ipiv_dev);
        device_free(queue, scratchpad_dev);
    }

    std::vector<fp> U(n * n);
    std::vector<fp> Uk(n * n);
    int64_t ldu = n;
    std::vector<fp> D(n * n);
    int64_t ldd = n;
    symmetric_to_full(uplo, n, A_initial, lda);
    bool result = true;

    for (int64_t d = 0; d < n; d++)
        U[d + d * ldu] = 1.0;

    if (uplo == oneapi::math::uplo::upper) {
        int64_t k = n - 1;
        while (k >= 0) {
            reference::laset('A', n, n, 0.0, 1.0, Uk.data(), ldu);
            if (ipiv[k] > 0) { /* 1x1 block case */

                auto piv = ipiv[k] - 1;
                for (int64_t i = 0; i < k; i++)
                    Uk[i + k * ldu] = A[i + k * lda];
                if (piv != k)
                    reference::swap(n, Uk.data() + (k + 0 * ldu), ldu, Uk.data() + (piv + 0 * ldu),
                                    ldu);
                auto U_temp = U;
                reference::gemm(oneapi::math::transpose::nontrans,
                                oneapi::math::transpose::nontrans, n, n, n, 1.0, U_temp.data(), ldu,
                                Uk.data(), ldu, 0.0, U.data(), ldu);

                D[k + k * ldd] = A[k + k * lda];
                k -= 1;
            }
            else { /* 2x2 block case */

                auto piv = -ipiv[k] - 1;
                for (int64_t i = 0; i < k - 1; i++) {
                    Uk[i + k * ldu] = A[i + k * lda];
                    Uk[i + (k - 1) * ldu] = A[i + (k - 1) * lda];
                }
                if (piv != k - 1)
                    reference::swap(n, Uk.data() + (k - 1 + 0 * ldu), ldu,
                                    Uk.data() + (piv + 0 * ldu), ldu);
                auto U_temp = U;
                reference::gemm(oneapi::math::transpose::nontrans,
                                oneapi::math::transpose::nontrans, n, n, n, 1.0, U_temp.data(), ldu,
                                Uk.data(), ldu, 0.0, U.data(), ldu);

                D[k + k * ldd] = A[k + k * lda];
                D[k - 1 + (k - 1) * ldd] = A[k - 1 + (k - 1) * lda];
                D[k - 1 + k * ldd] = A[k - 1 + k * lda];
                D[k + (k - 1) * ldd] = A[k - 1 + k * lda];
                k -= 2;
            }
        }
    }
    else {
        int64_t k = 0;
        while (k < n) {
            reference::laset('A', n, n, 0.0, 1.0, Uk.data(), ldu);
            if (ipiv[k] > 0) { /* 1x1 block case */

                auto piv = ipiv[k] - 1;
                for (int64_t i = k + 1; i < n; i++)
                    Uk[i + k * ldu] = A[i + k * lda];
                if (piv != k)
                    reference::swap(n, Uk.data() + (k + 0 * lda), ldu, Uk.data() + (piv + 0 * ldu),
                                    ldu);
                auto U_temp = U;
                reference::gemm(oneapi::math::transpose::nontrans,
                                oneapi::math::transpose::nontrans, n, n, n, 1.0, U_temp.data(), ldu,
                                Uk.data(), ldu, 0.0, U.data(), ldu);

                D[k + (k)*ldd] = A[k + (k)*lda];
                k += 1;
            }
            else { /* 2x2 block case */

                auto piv = -ipiv[k] - 1;
                for (int64_t i = k + 2; i < n; i++) {
                    Uk[i + k * ldu] = A[i + k * lda];
                    Uk[i + (k + 1) * ldu] = A[i + (k + 1) * lda];
                }
                if (piv != k)
                    reference::swap(n, Uk.data() + (k + 1 + 0 * ldu), ldu,
                                    Uk.data() + (piv + 0 * ldu), ldu);
                auto U_temp = U;
                reference::gemm(oneapi::math::transpose::nontrans,
                                oneapi::math::transpose::nontrans, n, n, n, 1.0, U_temp.data(), ldu,
                                Uk.data(), ldu, 0.0, U.data(), ldu);

                D[k + k * ldd] = A[k + k * lda];
                D[k + 1 + (k + 1) * ldd] = A[k + 1 + (k + 1) * lda];
                D[k + 1 + k * ldd] = A[k + 1 + k * lda];
                D[k + (k + 1) * ldd] = A[k + 1 + k * lda];
                k += 2;
            }
        }
    }

    /* |A - UDU'| < |A| O(eps) */
    std::vector<fp> UD(n * n);
    int64_t ldud = n;
    reference::gemm(oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, n, n, n,
                    1.0, U.data(), ldu, D.data(), ldd, 0.0, UD.data(), ldud);

    std::vector<fp> UDU(n * n);
    int64_t ldudu = n;
    reference::gemm(oneapi::math::transpose::nontrans, oneapi::math::transpose::trans, n, n, n, 1.0,
                    UD.data(), ldud, U.data(), ldu, 0.0, UDU.data(), ldudu);

    if (!rel_mat_err_check(n, n, UDU, ldudu, A_initial, lda)) {
        test_log::lout << "Factorization check failed" << std::endl;
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
    std::vector<int64_t> ipiv(n);
    rand_symmetric_matrix(seed, uplo, n, A, lda);

    std::vector<fp> A_initial = A;

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto ipiv_dev = device_alloc<data_T, int64_t>(queue, ipiv.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::math::lapack::sytrf_scratchpad_size<fp>(queue, uplo, n, lda);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::math::lapack::sytrf_scratchpad_size<fp>, uplo, n, lda);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, ipiv.data(), ipiv_dev, ipiv.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event =
            oneapi::math::lapack::sytrf(queue, uplo, n, A_dev, lda, ipiv_dev, scratchpad_dev,
                                        scratchpad_size, std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::math::lapack::sytrf, uplo, n, A_dev,
                                  lda, ipiv_dev, scratchpad_dev, scratchpad_size,
                                  std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, ipiv_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY(Sytrf);
INSTANTIATE_GTEST_SUITE_DEPENDENCY(Sytrf);
