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
1 4 15 5 123 22 400 3 27182
)";

template <typename data_T>
bool accuracy(const sycl::device& dev, oneapi::math::uplo uplo, int64_t n, int64_t nrhs,
              int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b, int64_t batch_size,
              uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(stride_a * batch_size);
    std::vector<fp> B_initial(stride_b * batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
        rand_pos_def_matrix(seed, uplo, n, A_initial, lda, i * stride_a);
        rand_matrix(seed, oneapi::math::transpose::nontrans, n, nrhs, B_initial, ldb, i * stride_b);
    }

    std::vector<fp> A = A_initial;
    std::vector<fp> B = B_initial;

    for (int64_t i = 0; i < batch_size; i++) {
        auto info = reference::potrf(uplo, n, A.data() + i * stride_a, lda);
        if (0 != info) {
            test_log::lout << "batch routine index " << i
                           << ": reference potrf failed with info: " << info << std::endl;
            return false;
        }
    }

    /* Compute on device */
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto B_dev = device_alloc<data_T>(queue, B.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::math::lapack::potrs_batch_scratchpad_size<fp>(
            queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::math::lapack::potrs_batch_scratchpad_size<fp>, uplo, n,
            nrhs, lda, stride_a, ldb, stride_b, batch_size);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, B.data(), B_dev, B.size());
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::math::lapack::potrs_batch(queue, uplo, n, nrhs, A_dev, lda, stride_a, B_dev, ldb,
                                          stride_b, batch_size, scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_LAPACK_CT_SELECT(queue, oneapi::math::lapack::potrs_batch, uplo, n, nrhs, A_dev,
                                  lda, stride_a, B_dev, ldb, stride_b, batch_size, scratchpad_dev,
                                  scratchpad_size);
#endif
        queue.wait_and_throw();

        device_to_host_copy(queue, B_dev, B.data(), B.size());
        queue.wait_and_throw();

        device_free(queue, A_dev);
        device_free(queue, B_dev);
        device_free(queue, scratchpad_dev);
    }

    bool result = true;
    for (int64_t i = 0; i < batch_size; i++) {
        auto B_ = copy_vector(B, ldb * nrhs, i * stride_b);
        auto A_initial_ = copy_vector(A_initial, lda * n, i * stride_a);
        auto B_initial_ = copy_vector(B_initial, ldb * nrhs, i * stride_b);
        if (!check_potrs_accuracy(uplo, n, nrhs, B_, ldb, A_initial_, lda, B_initial_)) {
            test_log::lout << "batch routine index " << i << " failed" << std::endl;
            result = false;
        }
    }

    return result;
}

const char* dependency_input = R"(
1 1 1 1 1 1 1 1 1
)";

template <typename data_T>
bool usm_dependency(const sycl::device& dev, oneapi::math::uplo uplo, int64_t n, int64_t nrhs,
                    int64_t lda, int64_t stride_a, int64_t ldb, int64_t stride_b,
                    int64_t batch_size, uint64_t seed) {
    using fp = typename data_T_info<data_T>::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    /* Initialize */
    std::vector<fp> A_initial(stride_a * batch_size);
    std::vector<fp> B_initial(stride_b * batch_size);
    for (int64_t i = 0; i < batch_size; i++) {
        rand_pos_def_matrix(seed, uplo, n, A_initial, lda, i * stride_a);
        rand_matrix(seed, oneapi::math::transpose::nontrans, n, nrhs, B_initial, ldb, i * stride_b);
    }

    std::vector<fp> A = A_initial;
    std::vector<fp> B = B_initial;

    for (int64_t i = 0; i < batch_size; i++) {
        auto info = reference::potrf(uplo, n, A.data() + i * stride_a, lda);
        if (0 != info) {
            test_log::lout << "batch routine index " << i
                           << ": reference potrf failed with info: " << info << std::endl;
            return false;
        }
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev, async_error_handler };

        auto A_dev = device_alloc<data_T>(queue, A.size());
        auto B_dev = device_alloc<data_T>(queue, B.size());
#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::math::lapack::potrs_batch_scratchpad_size<fp>(
            queue, uplo, n, nrhs, lda, stride_a, ldb, stride_b, batch_size);
#else
        int64_t scratchpad_size;
        TEST_RUN_LAPACK_CT_SELECT(
            queue, scratchpad_size = oneapi::math::lapack::potrs_batch_scratchpad_size<fp>, uplo, n,
            nrhs, lda, stride_a, ldb, stride_b, batch_size);
#endif
        auto scratchpad_dev = device_alloc<data_T>(queue, scratchpad_size);

        host_to_device_copy(queue, A.data(), A_dev, A.size());
        host_to_device_copy(queue, B.data(), B_dev, B.size());
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependency(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::math::lapack::potrs_batch(
            queue, uplo, n, nrhs, A_dev, lda, stride_a, B_dev, ldb, stride_b, batch_size,
            scratchpad_dev, scratchpad_size, std::vector<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_LAPACK_CT_SELECT(queue, func_event = oneapi::math::lapack::potrs_batch, uplo, n,
                                  nrhs, A_dev, lda, stride_a, B_dev, ldb, stride_b, batch_size,
                                  scratchpad_dev, scratchpad_size,
                                  std::vector<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
        device_free(queue, A_dev);
        device_free(queue, B_dev);
        device_free(queue, scratchpad_dev);
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* anonymous namespace */

#include "lapack_gtest_suite.hpp"
INSTANTIATE_GTEST_SUITE_ACCURACY(PotrsBatchStride);
INSTANTIATE_GTEST_SUITE_DEPENDENCY(PotrsBatchStride);
