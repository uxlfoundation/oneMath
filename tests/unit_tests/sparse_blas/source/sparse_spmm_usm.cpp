/*******************************************************************************
* Copyright 2024 Intel Corporation
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
#include <iostream>
#include <vector>

#include "test_spmm.hpp"

extern std::vector<sycl::device*> devices;

namespace {

template <typename fpType, typename intType>
int test_spmm(sycl::device* dev, sycl::property_list queue_properties,
              sparse_matrix_format_t format, intType nrows_A, intType ncols_A, intType ncols_C,
              double density_A_matrix, oneapi::math::index_base index,
              oneapi::math::layout dense_matrix_layout, oneapi::math::transpose transpose_A,
              oneapi::math::transpose transpose_B, fpType alpha, fpType beta, intType ldb,
              intType ldc, oneapi::math::sparse::spmm_alg alg,
              oneapi::math::sparse::matrix_view A_view,
              const std::set<oneapi::math::sparse::matrix_property>& matrix_properties,
              bool reset_data, bool test_scalar_on_device) {
    sycl::queue main_queue(*dev, exception_handler_t(), queue_properties);

    if (require_square_matrix(A_view, matrix_properties)) {
        ncols_A = nrows_A;
        ncols_C = nrows_A;
        ldb = nrows_A;
        ldc = nrows_A;
    }
    auto [opa_nrows, opa_ncols] = swap_if_transposed<std::size_t>(transpose_A, nrows_A, ncols_A);
    auto [opb_nrows, opb_ncols] = swap_if_transposed<std::int64_t>(transpose_B, opa_ncols, ncols_C);
    intType indexing = (index == oneapi::math::index_base::zero) ? 0 : 1;
    const bool is_symmetric =
        matrix_properties.find(oneapi::math::sparse::matrix_property::symmetric) !=
        matrix_properties.cend();

    // Input matrix
    std::vector<intType> ia_host, ja_host;
    std::vector<fpType> a_host;
    intType nnz =
        generate_random_matrix<fpType, intType>(format, nrows_A, ncols_A, density_A_matrix,
                                                indexing, ia_host, ja_host, a_host, is_symmetric);

    // Input and output dense vectors
    std::vector<fpType> b_host, c_host;
    rand_matrix(b_host, dense_matrix_layout, opa_ncols, static_cast<std::size_t>(ncols_C),
                static_cast<std::size_t>(ldb), transpose_B);
    rand_matrix(c_host, dense_matrix_layout, opa_nrows, static_cast<std::size_t>(ncols_C),
                static_cast<std::size_t>(ldc));
    std::vector<fpType> c_ref_host(c_host);

    // Shuffle ordering of column indices/values to test sortedness
    shuffle_sparse_matrix_if_needed(format, matrix_properties, indexing, ia_host.data(),
                                    ja_host.data(), a_host.data(), nnz,
                                    static_cast<std::size_t>(nrows_A));

    auto ia_usm_uptr = malloc_device_uptr<intType>(main_queue, ia_host.size());
    auto ja_usm_uptr = malloc_device_uptr<intType>(main_queue, ja_host.size());
    auto a_usm_uptr = malloc_device_uptr<fpType>(main_queue, a_host.size());
    auto b_usm_uptr = malloc_device_uptr<fpType>(main_queue, b_host.size());
    auto c_usm_uptr = malloc_device_uptr<fpType>(main_queue, c_host.size());
    auto alpha_usm_uptr = malloc_device_uptr<fpType>(main_queue, 1);
    auto beta_usm_uptr = malloc_device_uptr<fpType>(main_queue, 1);

    intType* ia_usm = ia_usm_uptr.get();
    intType* ja_usm = ja_usm_uptr.get();
    fpType* a_usm = a_usm_uptr.get();
    fpType* b_usm = b_usm_uptr.get();
    fpType* c_usm = c_usm_uptr.get();

    std::vector<sycl::event> dependencies;
    // Copy host to device
    dependencies.push_back(
        main_queue.memcpy(ia_usm, ia_host.data(), ia_host.size() * sizeof(intType)));
    dependencies.push_back(
        main_queue.memcpy(ja_usm, ja_host.data(), ja_host.size() * sizeof(intType)));
    dependencies.push_back(main_queue.memcpy(a_usm, a_host.data(), a_host.size() * sizeof(fpType)));
    dependencies.push_back(main_queue.memcpy(b_usm, b_host.data(), b_host.size() * sizeof(fpType)));
    dependencies.push_back(main_queue.memcpy(c_usm, c_host.data(), c_host.size() * sizeof(fpType)));

    fpType* alpha_host_or_usm_ptr = &alpha;
    fpType* beta_host_or_usm_ptr = &beta;
    if (test_scalar_on_device) {
        dependencies.push_back(main_queue.memcpy(alpha_usm_uptr.get(), &alpha, sizeof(fpType)));
        dependencies.push_back(main_queue.memcpy(beta_usm_uptr.get(), &beta, sizeof(fpType)));
        alpha_host_or_usm_ptr = alpha_usm_uptr.get();
        beta_host_or_usm_ptr = beta_usm_uptr.get();
    }

    sycl::event ev_copy, ev_spmm;
    oneapi::math::sparse::matrix_handle_t A_handle = nullptr;
    oneapi::math::sparse::dense_matrix_handle_t B_handle = nullptr;
    oneapi::math::sparse::dense_matrix_handle_t C_handle = nullptr;
    oneapi::math::sparse::spmm_descr_t descr = nullptr;
    std::unique_ptr<std::uint8_t, UsmDeleter> workspace_usm(nullptr, UsmDeleter(main_queue));
    try {
        init_sparse_matrix(main_queue, format, &A_handle, nrows_A, ncols_A, nnz, index, ia_usm,
                           ja_usm, a_usm);
        for (auto property : matrix_properties) {
            CALL_RT_OR_CT(oneapi::math::sparse::set_matrix_property, main_queue, A_handle,
                          property);
        }
        CALL_RT_OR_CT(oneapi::math::sparse::init_dense_matrix, main_queue, &B_handle, opb_nrows,
                      opb_ncols, ldb, dense_matrix_layout, b_usm);
        CALL_RT_OR_CT(oneapi::math::sparse::init_dense_matrix, main_queue, &C_handle,
                      static_cast<std::int64_t>(opa_nrows), ncols_C, ldc, dense_matrix_layout,
                      c_usm);

        CALL_RT_OR_CT(oneapi::math::sparse::init_spmm_descr, main_queue, &descr);

        std::size_t workspace_size = 0;
        CALL_RT_OR_CT(oneapi::math::sparse::spmm_buffer_size, main_queue, transpose_A, transpose_B,
                      &alpha, A_view, A_handle, B_handle, &beta, C_handle, alg, descr,
                      workspace_size);
        workspace_usm = malloc_device_uptr<std::uint8_t>(main_queue, workspace_size);

        sycl::event ev_opt;
        CALL_RT_OR_CT(ev_opt = oneapi::math::sparse::spmm_optimize, main_queue, transpose_A,
                      transpose_B, &alpha, A_view, A_handle, B_handle, &beta, C_handle, alg, descr,
                      workspace_usm.get(), dependencies);

        CALL_RT_OR_CT(ev_spmm = oneapi::math::sparse::spmm, main_queue, transpose_A, transpose_B,
                      &alpha, A_view, A_handle, B_handle, &beta, C_handle, alg, descr, { ev_opt });

        if (reset_data) {
            intType reset_nnz = generate_random_matrix<fpType, intType>(
                format, nrows_A, ncols_A, density_A_matrix, indexing, ia_host, ja_host, a_host,
                is_symmetric);
            shuffle_sparse_matrix_if_needed(format, matrix_properties, indexing, ia_host.data(),
                                            ja_host.data(), a_host.data(), reset_nnz,
                                            static_cast<std::size_t>(nrows_A));
            ev_spmm.wait_and_throw();
            if (reset_nnz > nnz) {
                ia_usm_uptr = malloc_device_uptr<intType>(main_queue, ia_host.size());
                ja_usm_uptr = malloc_device_uptr<intType>(main_queue, ja_host.size());
                a_usm_uptr = malloc_device_uptr<fpType>(main_queue, a_host.size());
                ia_usm = ia_usm_uptr.get();
                ja_usm = ja_usm_uptr.get();
                a_usm = a_usm_uptr.get();
            }
            nnz = reset_nnz;

            dependencies.clear();
            dependencies.push_back(main_queue.memcpy(ia_usm, ia_host.data(),
                                                     ia_host.size() * sizeof(intType), ev_spmm));
            dependencies.push_back(main_queue.memcpy(ja_usm, ja_host.data(),
                                                     ja_host.size() * sizeof(intType), ev_spmm));
            dependencies.push_back(
                main_queue.memcpy(a_usm, a_host.data(), a_host.size() * sizeof(fpType), ev_spmm));
            dependencies.push_back(
                main_queue.memcpy(c_usm, c_host.data(), c_host.size() * sizeof(fpType), ev_spmm));
            set_matrix_data(main_queue, format, A_handle, nrows_A, ncols_A, nnz, index, ia_usm,
                            ja_usm, a_usm);

            std::size_t workspace_size_2 = 0;
            CALL_RT_OR_CT(oneapi::math::sparse::spmm_buffer_size, main_queue, transpose_A,
                          transpose_B, &alpha, A_view, A_handle, B_handle, &beta, C_handle, alg,
                          descr, workspace_size_2);
            if (workspace_size_2 > workspace_size) {
                workspace_usm = malloc_device_uptr<std::uint8_t>(main_queue, workspace_size_2);
            }

            CALL_RT_OR_CT(ev_opt = oneapi::math::sparse::spmm_optimize, main_queue, transpose_A,
                          transpose_B, &alpha, A_view, A_handle, B_handle, &beta, C_handle, alg,
                          descr, workspace_usm.get(), dependencies);

            CALL_RT_OR_CT(ev_spmm = oneapi::math::sparse::spmm, main_queue, transpose_A,
                          transpose_B, &alpha, A_view, A_handle, B_handle, &beta, C_handle, alg,
                          descr, { ev_opt });
        }

        ev_copy = main_queue.memcpy(c_host.data(), c_usm, c_host.size() * sizeof(fpType), ev_spmm);
    }
    catch (const sycl::exception& e) {
        std::cout << "Caught synchronous SYCL exception during sparse SPMM:\n"
                  << e.what() << std::endl;
        print_error_code(e);
        return 0;
    }
    catch (const oneapi::math::unimplemented& e) {
        wait_and_free_handles(main_queue, A_handle, B_handle, C_handle);
        if (descr) {
            sycl::event ev_release_descr;
            CALL_RT_OR_CT(ev_release_descr = oneapi::math::sparse::release_spmm_descr, main_queue,
                          descr);
            ev_release_descr.wait();
        }
        return test_skipped;
    }
    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of sparse SPMM:\n" << error.what() << std::endl;
        return 0;
    }
    sycl::event ev_release_descr;
    CALL_RT_OR_CT(ev_release_descr = oneapi::math::sparse::release_spmm_descr, main_queue, descr,
                  { ev_spmm });
    ev_release_descr.wait_and_throw();
    free_handles(main_queue, { ev_spmm }, A_handle, B_handle, C_handle);

    // Compute reference.
    prepare_reference_spmm_data(format, ia_host.data(), ja_host.data(), a_host.data(), nrows_A,
                                ncols_A, ncols_C, nnz, indexing, dense_matrix_layout, transpose_A,
                                transpose_B, alpha, beta, ldb, ldc, b_host.data(), A_view,
                                c_ref_host.data());

    // Compare the results of reference implementation and DPC++ implementation.
    ev_copy.wait_and_throw();
    bool valid = check_equal_vector(c_host, c_ref_host);

    return static_cast<int>(valid);
}

class SparseSpmmUsmTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(SparseSpmmUsmTests, RealSinglePrecision) {
    using fpType = float;
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spmm<fpType, int32_t>, test_spmm<fpType, std::int64_t>, GetParam(),
                        num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpmmUsmTests, RealDoublePrecision) {
    using fpType = double;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spmm<fpType, int32_t>, test_spmm<fpType, std::int64_t>, GetParam(),
                        num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpmmUsmTests, ComplexSinglePrecision) {
    using fpType = std::complex<float>;
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spmm<fpType, int32_t>, test_spmm<fpType, std::int64_t>, GetParam(),
                        num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpmmUsmTests, ComplexDoublePrecision) {
    using fpType = std::complex<double>;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spmm<fpType, int32_t>, test_spmm<fpType, std::int64_t>, GetParam(),
                        num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(SparseSpmmUsmTestSuite, SparseSpmmUsmTests, testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
