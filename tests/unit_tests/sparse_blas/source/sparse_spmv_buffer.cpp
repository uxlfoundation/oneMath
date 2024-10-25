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

#include "test_spmv.hpp"

extern std::vector<sycl::device*> devices;

namespace {

template <typename fpType, typename intType>
int test_spmv(sycl::device* dev, sycl::property_list queue_properties,
              sparse_matrix_format_t format, intType nrows_A, intType ncols_A,
              double density_A_matrix, oneapi::mkl::index_base index,
              oneapi::mkl::transpose transpose_val, fpType alpha, fpType beta,
              oneapi::mkl::sparse::spmv_alg alg, oneapi::mkl::sparse::matrix_view A_view,
              const std::set<oneapi::mkl::sparse::matrix_property>& matrix_properties,
              bool reset_data, bool test_scalar_on_device) {
    if (test_scalar_on_device) {
        // Scalars on the device is not planned to be supported with the buffer API
        return 1;
    }
    sycl::queue main_queue(*dev, exception_handler_t(), queue_properties);

    if (require_square_matrix(A_view, matrix_properties)) {
        ncols_A = nrows_A;
    }
    auto [opa_nrows, opa_ncols] = swap_if_transposed<std::size_t>(transpose_val, nrows_A, ncols_A);
    intType indexing = (index == oneapi::mkl::index_base::zero) ? 0 : 1;
    const bool is_symmetric =
        matrix_properties.find(oneapi::mkl::sparse::matrix_property::symmetric) !=
        matrix_properties.cend();

    // Input matrix
    std::vector<intType> ia_host, ja_host;
    std::vector<fpType> a_host;
    intType nnz =
        generate_random_matrix<fpType, intType>(format, nrows_A, ncols_A, density_A_matrix,
                                                indexing, ia_host, ja_host, a_host, is_symmetric);

    // Input and output dense vectors
    // The input `x` and the input-output `y` are both initialized to random values on host and device.
    std::vector<fpType> x_host, y_host;
    rand_vector(x_host, opa_ncols);
    rand_vector(y_host, opa_nrows);
    std::vector<fpType> y_ref_host(y_host);

    // Shuffle ordering of column indices/values to test sortedness
    shuffle_sparse_matrix_if_needed(format, matrix_properties, indexing, ia_host.data(),
                                    ja_host.data(), a_host.data(), nnz,
                                    static_cast<std::size_t>(nrows_A));

    auto ia_buf = make_buffer(ia_host);
    auto ja_buf = make_buffer(ja_host);
    auto a_buf = make_buffer(a_host);
    auto x_buf = make_buffer(x_host);
    auto y_buf = make_buffer(y_host);

    oneapi::mkl::sparse::matrix_handle_t A_handle = nullptr;
    oneapi::mkl::sparse::dense_vector_handle_t x_handle = nullptr;
    oneapi::mkl::sparse::dense_vector_handle_t y_handle = nullptr;
    oneapi::mkl::sparse::spmv_descr_t descr = nullptr;
    try {
        init_sparse_matrix(main_queue, format, &A_handle, nrows_A, ncols_A, nnz, index, ia_buf,
                           ja_buf, a_buf);
        for (auto property : matrix_properties) {
            CALL_RT_OR_CT(oneapi::mkl::sparse::set_matrix_property, main_queue, A_handle, property);
        }
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_dense_vector, main_queue, &x_handle,
                      static_cast<std::int64_t>(x_host.size()), x_buf);
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_dense_vector, main_queue, &y_handle,
                      static_cast<std::int64_t>(y_host.size()), y_buf);

        CALL_RT_OR_CT(oneapi::mkl::sparse::init_spmv_descr, main_queue, &descr);

        std::size_t workspace_size = 0;
        CALL_RT_OR_CT(oneapi::mkl::sparse::spmv_buffer_size, main_queue, transpose_val, &alpha,
                      A_view, A_handle, x_handle, &beta, y_handle, alg, descr, workspace_size);
        sycl::buffer<std::uint8_t, 1> workspace_buf((sycl::range<1>(workspace_size)));

        CALL_RT_OR_CT(oneapi::mkl::sparse::spmv_optimize, main_queue, transpose_val, &alpha, A_view,
                      A_handle, x_handle, &beta, y_handle, alg, descr, workspace_buf);

        CALL_RT_OR_CT(oneapi::mkl::sparse::spmv, main_queue, transpose_val, &alpha, A_view,
                      A_handle, x_handle, &beta, y_handle, alg, descr);

        if (reset_data) {
            intType reset_nnz = generate_random_matrix<fpType, intType>(
                format, nrows_A, ncols_A, density_A_matrix, indexing, ia_host, ja_host, a_host,
                is_symmetric);
            shuffle_sparse_matrix_if_needed(format, matrix_properties, indexing, ia_host.data(),
                                            ja_host.data(), a_host.data(), reset_nnz,
                                            static_cast<std::size_t>(nrows_A));
            if (reset_nnz > nnz) {
                ia_buf = make_buffer(ia_host);
                ja_buf = make_buffer(ja_host);
                a_buf = make_buffer(a_host);
            }
            else {
                copy_host_to_buffer(main_queue, ia_host, ia_buf);
                copy_host_to_buffer(main_queue, ja_host, ja_buf);
                copy_host_to_buffer(main_queue, a_host, a_buf);
            }
            copy_host_to_buffer(main_queue, y_ref_host, y_buf);
            nnz = reset_nnz;
            set_matrix_data(main_queue, format, A_handle, nrows_A, ncols_A, nnz, index, ia_buf,
                            ja_buf, a_buf);

            std::size_t workspace_size_2 = 0;
            CALL_RT_OR_CT(oneapi::mkl::sparse::spmv_buffer_size, main_queue, transpose_val, &alpha,
                          A_view, A_handle, x_handle, &beta, y_handle, alg, descr,
                          workspace_size_2);
            if (workspace_size_2 > workspace_size) {
                workspace_buf = sycl::buffer<std::uint8_t, 1>((sycl::range<1>(workspace_size_2)));
            }

            CALL_RT_OR_CT(oneapi::mkl::sparse::spmv_optimize, main_queue, transpose_val, &alpha,
                          A_view, A_handle, x_handle, &beta, y_handle, alg, descr, workspace_buf);

            CALL_RT_OR_CT(oneapi::mkl::sparse::spmv, main_queue, transpose_val, &alpha, A_view,
                          A_handle, x_handle, &beta, y_handle, alg, descr);
        }
    }
    catch (const sycl::exception& e) {
        std::cout << "Caught synchronous SYCL exception during sparse SPMV:\n"
                  << e.what() << std::endl;
        print_error_code(e);
        return 0;
    }
    catch (const oneapi::mkl::unimplemented& e) {
        wait_and_free_handles(main_queue, A_handle, x_handle, y_handle);
        if (descr) {
            sycl::event ev_release_descr;
            CALL_RT_OR_CT(ev_release_descr = oneapi::mkl::sparse::release_spmv_descr, main_queue,
                          descr);
            ev_release_descr.wait();
        }
        return test_skipped;
    }
    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of sparse SPMV:\n" << error.what() << std::endl;
        return 0;
    }
    CALL_RT_OR_CT(oneapi::mkl::sparse::release_spmv_descr, main_queue, descr);
    free_handles(main_queue, A_handle, x_handle, y_handle);

    // Compute reference.
    prepare_reference_spmv_data(format, ia_host.data(), ja_host.data(), a_host.data(), nrows_A,
                                ncols_A, nnz, indexing, transpose_val, alpha, beta, x_host.data(),
                                A_view, y_ref_host.data());

    // Compare the results of reference implementation and DPC++ implementation.
    auto y_acc = y_buf.get_host_access(sycl::read_only);
    bool valid = check_equal_vector(y_acc, y_ref_host);

    return static_cast<int>(valid);
}

class SparseSpmvBufferTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(SparseSpmvBufferTests, RealSinglePrecision) {
    using fpType = float;
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spmv<fpType, int32_t>, test_spmv<fpType, std::int64_t>, GetParam(),
                        num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpmvBufferTests, RealDoublePrecision) {
    using fpType = double;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spmv<fpType, int32_t>, test_spmv<fpType, std::int64_t>, GetParam(),
                        num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpmvBufferTests, ComplexSinglePrecision) {
    using fpType = std::complex<float>;
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spmv<fpType, int32_t>, test_spmv<fpType, std::int64_t>, GetParam(),
                        num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpmvBufferTests, ComplexDoublePrecision) {
    using fpType = std::complex<double>;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spmv<fpType, int32_t>, test_spmv<fpType, std::int64_t>, GetParam(),
                        num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(SparseSpmvBufferTestSuite, SparseSpmvBufferTests,
                         testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
