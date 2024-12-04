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
1 1 0 27 33 31 27182
2 1 0 27 33 31 27182
3 1 0 27 33 31 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, int64_t itype, oneapi::math::job jobz,
              oneapi::math::uplo uplo, int64_t n, int64_t lda, int64_t ldb, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda * n);
    rand_hermitian_matrix(seed, uplo, n, A, lda);
    std::vector<fp> B(ldb * n);
    rand_pos_def_matrix(seed, uplo, n, B, ldb);

    std::vector<fp> A_initial = A;
    std::vector<fp> B_initial = B;
    std::vector<fp_real> w(n);

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };
        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto B_dev = device_alloc<data_T>(queue, B.size());
        auto w_dev = device_alloc<data_T, fp_real>(queue, w.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::math::lapack::hegvd_scratchpad_size<fp>(queue, itype, jobz, uplo, n, lda, ldb);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(queue,
                                  scratchpad_size = oneapi::math::lapack::hegvd_scratchpad_size<fp>,
                                  itype, jobz, uplo, n, lda, ldb);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, B.data(), B_dev, B.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::math::lapack::hegvd(queue, itype, jobz, uplo, n, A_dev, lda, B_dev, ldb, w_dev,
                                    scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::math::lapack::hegvd, itype, jobz, uplo, n, A_dev,
                                  lda, B_dev, ldb, w_dev, scratchpad_dev, scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, A_dev, A.data(), A.size());
        device_to_host_copy(queue, B_dev, B.data(), B.size());
        device_to_host_copy(queue, w_dev, w.data(), w.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, B_dev);
        device_free(queue, w_dev);
        device_free(queue, scratchpad_dev);
    }

    const auto& Z = A;
    auto ldz = lda;
    const auto& D = w;
    hermitian_to_full(uplo, n, A_initial, lda);
    hermitian_to_full(uplo, n, B_initial, ldb);
    bool result = true;

    /* |D_ref - D| < |D_ref| O(eps) */
    std::vector<fp_real> D_ref(n);
    reference::hegvd(itype, oneapi::math::job::novec, uplo, n, std::vector<fp>(A_initial).data(),
                     lda, std::vector<fp>(B_initial).data(), ldb, D_ref.data());
    if (!rel_vec_err_check(n, D_ref, D, 10.0)) {
        test_log::lout << "Eigenvalue check failed" << std::endl;
        result = false;
    }

    if (oneapi::math::job::vec == jobz) {
        if (itype == 1) {
            /* |A Z - B Z D| < |A Z| O(eps) */
            std::vector<fp> AZ(n * n);
            int64_t ldaz = n;
            reference::gemm(oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, n,
                            n, n, 1.0, A_initial.data(), lda, Z.data(), ldz, 0.0, AZ.data(), ldaz);

            std::vector<fp> BZ(n * n);
            int64_t ldbz = n;
            reference::gemm(oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, n,
                            n, n, 1.0, B_initial.data(), ldb, Z.data(), ldz, 0.0, BZ.data(), ldbz);

            std::vector<fp> BZD(n * n);
            int64_t ldbzd = n;
            for (int64_t col = 0; col < n; col++)
                for (int64_t row = 0; row < n; row++)
                    BZD[row + col * ldbzd] = BZ[row + col * ldbz] * D[col];

            if (!rel_mat_err_check(n, n, AZ, ldaz, BZD, ldbzd)) {
                test_log::lout << "Factorization check failed" << std::endl;
                result = false;
            }

            /* |I - Z' B Z| < n O(eps) */
            std::vector<fp> ZBZ(n * n);
            int64_t ldzbz = n;
            reference::gemm(oneapi::math::transpose::conjtrans, oneapi::math::transpose::nontrans,
                            n, n, n, 1.0, Z.data(), ldz, BZ.data(), ldbz, 0.0, ZBZ.data(), ldzbz);
            if (!rel_id_err_check(n, ZBZ, ldzbz)) {
                test_log::lout << "Orthogonality check failed" << std::endl;
                result = false;
            }
        }
        else if (itype == 2) {
            /* |A B Z - Z D| < |A B Z| O(eps) */
            std::vector<fp> BZ(n * n);
            int64_t ldbz = n;
            reference::gemm(oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, n,
                            n, n, 1.0, B_initial.data(), ldb, Z.data(), ldz, 0.0, BZ.data(), ldbz);

            std::vector<fp> ABZ(n * n);
            int64_t ldabz = n;
            reference::gemm(oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, n,
                            n, n, 1.0, A_initial.data(), lda, BZ.data(), ldbz, 0.0, ABZ.data(),
                            ldabz);

            std::vector<fp> ZD(n * n);
            int64_t ldzd = n;
            for (int64_t col = 0; col < n; col++)
                for (int64_t row = 0; row < n; row++)
                    ZD[row + col * ldzd] = Z[row + col * ldz] * D[col];

            if (!rel_mat_err_check(n, n, ABZ, ldabz, ZD, ldbz)) {
                test_log::lout << "Factorization check failed" << std::endl;
                result = false;
            }

            /* |I - Z' B Z| < n O(eps) */
            std::vector<fp> ZBZ(n * n);
            int64_t ldzbz = n;
            reference::gemm(oneapi::math::transpose::conjtrans, oneapi::math::transpose::nontrans,
                            n, n, n, 1.0, Z.data(), ldz, BZ.data(), ldbz, 0.0, ZBZ.data(), ldzbz);
            if (!rel_id_err_check(n, ZBZ, ldzbz)) {
                test_log::lout << "Orthogonality check failed" << std::endl;
                result = false;
            }
        }
        else {
            /* |A Z - B^-1 Z D| < |A Z| O(eps) */
            /* C = B^-1 Z */
            std::vector<fp> AZ(n * n);
            int64_t ldaz = n;
            reference::gemm(oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans, n,
                            n, n, 1.0, A_initial.data(), lda, Z.data(), ldz, 0.0, AZ.data(), ldaz);

            std::vector<fp> C(n * n);
            int64_t ldc = n;
            reference::lacpy('A', n, n, Z.data(), ldz, C.data(), ldc);
            auto info = reference::potrs(uplo, n, n, B.data(), ldb, C.data(), ldc);
            if (0 != info) {
                test_log::lout << "reference potrs failed with info = " << info << std::endl;
                return false;
            }

            std::vector<fp> CD(n * n);
            int64_t ldcd = n;
            for (int64_t col = 0; col < n; col++)
                for (int64_t row = 0; row < n; row++)
                    CD[row + col * ldcd] = C[row + col * ldc] * D[col];

            if (!rel_mat_err_check(n, n, AZ, ldaz, CD, ldcd)) {
                test_log::lout << "Factorization check failed" << std::endl;
                result = false;
            }

            /* |I - Z' B^-1 Z| = |I - Z' C| < n O(eps) */
            std::vector<fp> ZhC(n * n);
            int64_t ldzhc = n;
            reference::gemm(oneapi::math::transpose::conjtrans, oneapi::math::transpose::nontrans,
                            n, n, n, 1.0, Z.data(), ldz, C.data(), ldc, 0.0, ZhC.data(), ldzhc);
            if (!rel_id_err_check(n, ZhC, ldzhc)) {
                test_log::lout << "Orthogonality check failed" << std::endl;
                result = false;
            }
        }
    }
    return result;
}

const char* dependency_input = R"(
1 1 1 1 1 1 1
)";

template <typename data_T>
bool usm_dependency(const sycl::device& dev, int64_t itype, oneapi::math::job jobz,
                    oneapi::math::uplo uplo, int64_t n, int64_t lda, int64_t ldb, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A(lda * n);
    rand_hermitian_matrix(seed, uplo, n, A, lda);
    std::vector<fp> B(ldb * n);
    rand_pos_def_matrix(seed, uplo, n, B, ldb);

    std::vector<fp> A_initial = A;
    std::vector<fp> B_initial = B;
    std::vector<fp_real> w(n);

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };
        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto B_dev = device_alloc<data_T>(queue, B.size());
        auto w_dev = device_alloc<data_T, fp_real>(queue, w.size());
#ifdef CALL_RT_API
        const auto scratchpad_size =
            oneapi::math::lapack::hegvd_scratchpad_size<fp>(queue, itype, jobz, uplo, n, lda, ldb);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(queue,
                                  scratchpad_size = oneapi::math::lapack::hegvd_scratchpad_size<fp>,
                                  itype, jobz, uplo, n, lda, ldb);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, B.data(), B_dev, B.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::math::lapack::hegvd(
            queue, itype, jobz, uplo, n, A_dev, lda, B_dev, ldb, w_dev, scratchpad_dev,
            scratchpad_size, std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::math::lapack::hegvd, itype, jobz,
                                  uplo, n, A_dev, lda, B_dev, ldb, w_dev, scratchpad_dev,
                                  scratchpad_size, std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, B_dev);
        device_free(queue, w_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY_COMPLEX(Hegvd);
INSTANTIATE_GTEST_SUITE_DEPENDENCY_COMPLEX(Hegvd);
