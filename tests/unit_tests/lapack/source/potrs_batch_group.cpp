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
#include <list>
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "lapack_common.hpp"
#include "lapack_test_controller.hpp"
#include "lapack_accuracy_checks.hpp"
#include "reference_lapack_wrappers.hpp"
#include "test_helper.hpp"

namespace {

const char* accuracy_input = R"(
27182
)";

template <typename fp>
bool accuracy(const sycl::device& dev, uint64_t seed) {
    using fp_real = typename complex_info<fp>::real_type;

    /* Test Parameters */
    std::vector<oneapi::mkl::uplo> uplo_vec = { oneapi::mkl::uplo::upper,
                                                oneapi::mkl::uplo::lower };
    std::vector<int64_t> n_vec = { 4, 5 };
    std::vector<int64_t> nrhs_vec = { 9, 6 };
    std::vector<int64_t> lda_vec = { 6, 6 };
    std::vector<int64_t> ldb_vec = { 9, 9 };
    std::vector<int64_t> group_sizes_vec = { 2, 2 };

    int64_t group_count = group_sizes_vec.size();
    int64_t batch_size = std::accumulate(group_sizes_vec.begin(), group_sizes_vec.end(), 0);

    std::list<std::vector<fp>> A_initial_list;
    std::list<std::vector<fp>> B_initial_list;
    std::list<std::vector<fp>> A_list;
    std::list<std::vector<fp>> B_list;

    for (int64_t group_id = 0; group_id < group_count; group_id++) {
        auto uplo = uplo_vec[group_id];
        auto n = n_vec[group_id];
        auto nrhs = nrhs_vec[group_id];
        auto lda = lda_vec[group_id];
        auto ldb = ldb_vec[group_id];
        auto group_size = group_sizes_vec[group_id];

        /* Allocate and Initialize on host */
        for (int64_t local_id = 0; local_id < group_size; local_id++) {
            A_initial_list.emplace_back(lda * n);
            auto& A_initial = A_initial_list.back();
            rand_pos_def_matrix(seed, uplo, n, A_initial, lda);

            A_list.emplace_back(A_initial);
            auto& A = A_list.back();

            B_initial_list.emplace_back(ldb * nrhs);
            auto& B_initial = B_initial_list.back();
            rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, nrhs, B_initial, lda);

            B_list.emplace_back(B_initial);
            auto& B = B_list.back();

            auto info = reference::potrf(uplo, n, A.data(), lda);
            if (info != 0) {
                global::log << "\tReference getrf failed with info = " << info << std::endl;
                return false;
            }
        }
    }

    /* Compute on device */
    {
        sycl::queue queue{ dev };

        std::list<std::vector<fp, sycl::usm_allocator<fp, sycl::usm::alloc::shared>>> A_dev_list;
        std::list<std::vector<fp, sycl::usm_allocator<fp, sycl::usm::alloc::shared>>> B_dev_list;
        std::vector<fp*> A_dev_ptrs(batch_size, nullptr);
        std::vector<fp*> B_dev_ptrs(batch_size, nullptr);

        /* Allocate on device */
        sycl::usm_allocator<fp, sycl::usm::alloc::shared> usm_fp_allocator{ queue.get_context(),
                                                                            dev };
        auto A_iter = A_list.begin();
        auto B_iter = B_list.begin();
        for (int64_t global_id = 0; global_id < batch_size; global_id++, A_iter++, B_iter++) {
            A_dev_list.emplace_back(A_iter->size(), usm_fp_allocator);
            B_dev_list.emplace_back(B_iter->size(), usm_fp_allocator);
        }

#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<fp>(
            queue, uplo_vec.data(), n_vec.data(), nrhs_vec.data(), lda_vec.data(), ldb_vec.data(),
            group_count, group_sizes_vec.data());
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue,
                           scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<fp>,
                           uplo_vec.data(), n_vec.data(), nrhs_vec.data(), lda_vec.data(),
                           ldb_vec.data(), group_count, group_sizes_vec.data());
#endif
        auto scratchpad_dev = device_alloc<fp>(queue, scratchpad_size);

        auto A_dev_iter = A_dev_list.begin();
        auto B_dev_iter = B_dev_list.begin();
        for (int64_t global_id = 0; global_id < batch_size;
             global_id++, A_dev_iter++, B_dev_iter++) {
            A_dev_ptrs[global_id] = A_dev_iter->data();
            B_dev_ptrs[global_id] = B_dev_iter->data();
        }

        A_iter = A_list.begin();
        B_iter = B_list.begin();
        for (int64_t global_id = 0; global_id < batch_size; global_id++, A_iter++, B_iter++) {
            host_to_device_copy(queue, A_iter->data(), A_dev_ptrs[global_id], A_iter->size());
            host_to_device_copy(queue, B_iter->data(), B_dev_ptrs[global_id], B_iter->size());
        }
        queue.wait_and_throw();

#ifdef CALL_RT_API
        oneapi::mkl::lapack::potrs_batch(queue, uplo_vec.data(), n_vec.data(), nrhs_vec.data(),
                                         A_dev_ptrs.data(), lda_vec.data(), B_dev_ptrs.data(),
                                         ldb_vec.data(), group_count, group_sizes_vec.data(),
                                         scratchpad_dev, scratchpad_size);
#else
        TEST_RUN_CT_SELECT(queue, oneapi::mkl::lapack::potrs_batch, uplo_vec.data(), n_vec.data(),
                           nrhs_vec.data(), A_dev_ptrs.data(), lda_vec.data(), B_dev_ptrs.data(),
                           ldb_vec.data(), group_count, group_sizes_vec.data(), scratchpad_dev,
                           scratchpad_size);
#endif
        queue.wait_and_throw();

        B_iter = B_list.begin();
        for (int64_t global_id = 0; global_id < batch_size; global_id++, B_iter++) {
            device_to_host_copy(queue, B_dev_ptrs[global_id], B_iter->data(), B_iter->size());
        }
        queue.wait_and_throw();
    }

    bool result = true;

    int64_t global_id = 0;
    auto A_iter = A_list.begin();
    auto B_iter = B_list.begin();
    auto A_initial_iter = A_initial_list.begin();
    auto B_initial_iter = B_initial_list.begin();
    for (int64_t group_id = 0; group_id < group_count; group_id++) {
        auto uplo = uplo_vec[group_id];
        auto n = n_vec[group_id];
        auto nrhs = nrhs_vec[group_id];
        auto lda = lda_vec[group_id];
        auto ldb = ldb_vec[group_id];
        auto group_size = group_sizes_vec[group_id];
        for (int64_t local_id = 0; local_id < group_size;
             local_id++, global_id++, A_iter++, B_iter++, A_initial_iter++, B_initial_iter++) {
            if (!check_potrs_accuracy(uplo, n, nrhs, A_iter->data(), lda, B_iter->data(), ldb,
                                      A_initial_iter->data(), B_initial_iter->data())) {
                global::log << "\tbatch routine (" << global_id << ", " << group_id << ", "
                            << local_id << ") (global_id, group_id, local_id) failed" << std::endl;
                result = false;
            }
        }
    }

    return result;
}

const char* dependency_input = R"(
1
)";

template <typename fp>
bool usm_dependency(const sycl::device& dev, uint64_t seed) {
    using fp_real = typename complex_info<fp>::real_type;

    /* Test Parameters */
    std::vector<oneapi::mkl::uplo> uplo_vec = { oneapi::mkl::uplo::upper };
    std::vector<int64_t> n_vec = { 1 };
    std::vector<int64_t> nrhs_vec = { 1 };
    std::vector<int64_t> lda_vec = { 1 };
    std::vector<int64_t> ldb_vec = { 1 };
    std::vector<int64_t> group_sizes_vec = { 1 };

    int64_t group_count = group_sizes_vec.size();
    int64_t batch_size = std::accumulate(group_sizes_vec.begin(), group_sizes_vec.end(), 0);

    std::list<std::vector<fp>> A_initial_list;
    std::list<std::vector<fp>> B_initial_list;
    std::list<std::vector<fp>> A_list;
    std::list<std::vector<fp>> B_list;

    for (int64_t group_id = 0; group_id < group_count; group_id++) {
        auto uplo = uplo_vec[group_id];
        auto n = n_vec[group_id];
        auto nrhs = nrhs_vec[group_id];
        auto lda = lda_vec[group_id];
        auto ldb = ldb_vec[group_id];
        auto group_size = group_sizes_vec[group_id];

        /* Allocate and Initialize on host */
        for (int64_t local_id = 0; local_id < group_size; local_id++) {
            A_initial_list.emplace_back(lda * n);
            auto& A_initial = A_initial_list.back();
            rand_pos_def_matrix(seed, uplo, n, A_initial, lda);

            A_list.emplace_back(A_initial);
            auto& A = A_list.back();

            B_initial_list.emplace_back(ldb * nrhs);
            auto& B_initial = B_initial_list.back();
            rand_matrix(seed, oneapi::mkl::transpose::nontrans, n, nrhs, B_initial, lda);

            B_list.emplace_back(B_initial);
            auto& B = B_list.back();

            auto info = reference::potrf(uplo, n, A.data(), lda);
            if (info != 0) {
                global::log << "\tReference getrf failed with info = " << info << std::endl;
                return false;
            }
        }
    }

    /* Compute on device */
    bool result;
    {
        sycl::queue queue{ dev };

        std::list<std::vector<fp, sycl::usm_allocator<fp, sycl::usm::alloc::shared>>> A_dev_list;
        std::list<std::vector<fp, sycl::usm_allocator<fp, sycl::usm::alloc::shared>>> B_dev_list;
        std::vector<fp*> A_dev_ptrs(batch_size, nullptr);
        std::vector<fp*> B_dev_ptrs(batch_size, nullptr);

        /* Allocate on device */
        sycl::usm_allocator<fp, sycl::usm::alloc::shared> usm_fp_allocator{ queue.get_context(),
                                                                            dev };
        auto A_iter = A_list.begin();
        auto B_iter = B_list.begin();
        for (int64_t global_id = 0; global_id < batch_size; global_id++, A_iter++, B_iter++) {
            A_dev_list.emplace_back(A_iter->size(), usm_fp_allocator);
            B_dev_list.emplace_back(B_iter->size(), usm_fp_allocator);
        }

#ifdef CALL_RT_API
        const auto scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<fp>(
            queue, uplo_vec.data(), n_vec.data(), nrhs_vec.data(), lda_vec.data(), ldb_vec.data(),
            group_count, group_sizes_vec.data());
#else
        int64_t scratchpad_size;
        TEST_RUN_CT_SELECT(queue,
                           scratchpad_size = oneapi::mkl::lapack::potrs_batch_scratchpad_size<fp>,
                           uplo_vec.data(), n_vec.data(), nrhs_vec.data(), lda_vec.data(),
                           ldb_vec.data(), group_count, group_sizes_vec.data());
#endif
        auto scratchpad_dev = device_alloc<fp>(queue, scratchpad_size);

        auto A_dev_iter = A_dev_list.begin();
        auto B_dev_iter = B_dev_list.begin();
        for (int64_t global_id = 0; global_id < batch_size;
             global_id++, A_dev_iter++, B_dev_iter++) {
            A_dev_ptrs[global_id] = A_dev_iter->data();
            B_dev_ptrs[global_id] = B_dev_iter->data();
        }

        A_iter = A_list.begin();
        B_iter = B_list.begin();
        for (int64_t global_id = 0; global_id < batch_size; global_id++, A_iter++, B_iter++) {
            host_to_device_copy(queue, A_iter->data(), A_dev_ptrs[global_id], A_iter->size());
            host_to_device_copy(queue, B_iter->data(), B_dev_ptrs[global_id], B_iter->size());
        }
        queue.wait_and_throw();

        /* Check dependency handling */
        auto in_event = create_dependent_event(queue);
#ifdef CALL_RT_API
        sycl::event func_event = oneapi::mkl::lapack::potrs_batch(
            queue, uplo_vec.data(), n_vec.data(), nrhs_vec.data(), A_dev_ptrs.data(),
            lda_vec.data(), B_dev_ptrs.data(), ldb_vec.data(), group_count, group_sizes_vec.data(),
            scratchpad_dev, scratchpad_size, sycl::vector_class<sycl::event>{ in_event });
#else
        sycl::event func_event;
        TEST_RUN_CT_SELECT(queue, sycl::event func_event = oneapi::mkl::lapack::potrs_batch,
                           uplo_vec.data(), n_vec.data(), nrhs_vec.data(), A_dev_ptrs.data(),
                           lda_vec.data(), B_dev_ptrs.data(), ldb_vec.data(), group_count,
                           group_sizes_vec.data(), scratchpad_dev, scratchpad_size,
                           sycl::vector_class<sycl::event>{ in_event });
#endif
        result = check_dependency(queue, in_event, func_event);

        queue.wait_and_throw();
    }

    return result;
}

InputTestController<decltype(::accuracy<void>)> accuracy_controller{ accuracy_input };
InputTestController<decltype(::usm_dependency<void>)> dependency_controller{ dependency_input };

} /* unnamed namespace */

#ifdef STANDALONE
int main() {
    sycl::device dev = sycl::device{ sycl::host_selector{} };
    int64_t res = 0;
    res += !accuracy_controller.run(::accuracy<RealSinglePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<RealDoublePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexSinglePrecisionUsm>, dev);
    res += !accuracy_controller.run(::accuracy<ComplexDoublePrecisionUsm>, dev);
    res += !dependency_controller.run(::usm_dependency<RealSinglePrecisionUsm>, dev);
    res += !dependency_controller.run(::usm_dependency<RealDoublePrecisionUsm>, dev);
    res += !dependency_controller.run(::usm_dependency<ComplexSinglePrecisionUsm>, dev);
    res += !dependency_controller.run(::usm_dependency<ComplexDoublePrecisionUsm>, dev);
    return res;
}
#else
#include <gtest/gtest.h>
extern std::vector<sycl::device*> devices;
class PotrsBatchGroupTests : public ::testing::TestWithParam<sycl::device*> {};
INSTANTIATE_TEST_SUITE_P(PotrsBatchGroupTestSuite, PotrsBatchGroupTests,
                         ::testing::ValuesIn(devices), DeviceNamePrint());
RUN_SUITE_USM(PotrsBatchGroup)
#endif
