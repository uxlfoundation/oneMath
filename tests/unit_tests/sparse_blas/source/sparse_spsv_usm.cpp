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

#include "test_spsv.hpp"

extern std::vector<sycl::device*> devices;

namespace {

template <typename fpType, typename intType>
int test_spsv(sycl::device* dev, sycl::property_list queue_properties,
              sparse_matrix_format_t format, intType m, double density_A_matrix,
              oneapi::mkl::index_base index, oneapi::mkl::transpose transpose_val, fpType alpha,
              oneapi::mkl::sparse::spsv_alg alg, oneapi::mkl::sparse::matrix_view A_view,
              const std::set<oneapi::mkl::sparse::matrix_property>& matrix_properties,
              bool reset_data, bool test_scalar_on_device) {
    sycl::queue main_queue(*dev, exception_handler_t(), queue_properties);

    intType indexing = (index == oneapi::mkl::index_base::zero) ? 0 : 1;
    const std::size_t mu = static_cast<std::size_t>(m);
    const bool is_symmetric =
        matrix_properties.find(oneapi::mkl::sparse::matrix_property::symmetric) !=
        matrix_properties.cend();

    // Use a fixed seed for operations very sensitive to the input data
    std::srand(1);

    // Input matrix
    std::vector<intType> ia_host, ja_host;
    std::vector<fpType> a_host;
    // Set non-zero values to the diagonal, except if the matrix is viewed as a unit matrix.
    const bool require_diagonal =
        !(A_view.type_view == oneapi::mkl::sparse::matrix_descr::diagonal &&
          A_view.diag_view == oneapi::mkl::diag::unit);
    intType nnz =
        generate_random_matrix<fpType, intType>(format, m, m, density_A_matrix, indexing, ia_host,
                                                ja_host, a_host, is_symmetric, require_diagonal);

    // Input dense vector.
    // The input `x` is initialized to random values on host and device.
    std::vector<fpType> x_host;
    rand_vector(x_host, mu);

    // Output and reference dense vectors.
    // They are both initialized with a dummy value to catch more errors.
    std::vector<fpType> y_host(mu, -2.0f);
    std::vector<fpType> y_ref_host(y_host);

    // Shuffle ordering of column indices/values to test sortedness
    shuffle_sparse_matrix_if_needed(format, matrix_properties, indexing, ia_host.data(),
                                    ja_host.data(), a_host.data(), nnz, mu);

    auto ia_usm_uptr = malloc_device_uptr<intType>(main_queue, ia_host.size());
    auto ja_usm_uptr = malloc_device_uptr<intType>(main_queue, ja_host.size());
    auto a_usm_uptr = malloc_device_uptr<fpType>(main_queue, a_host.size());
    auto x_usm_uptr = malloc_device_uptr<fpType>(main_queue, x_host.size());
    auto y_usm_uptr = malloc_device_uptr<fpType>(main_queue, y_host.size());
    auto alpha_usm_uptr = malloc_device_uptr<fpType>(main_queue, 1);

    intType* ia_usm = ia_usm_uptr.get();
    intType* ja_usm = ja_usm_uptr.get();
    fpType* a_usm = a_usm_uptr.get();
    fpType* x_usm = x_usm_uptr.get();
    fpType* y_usm = y_usm_uptr.get();

    std::vector<sycl::event> dependencies;
    // Copy host to device
    dependencies.push_back(
        main_queue.memcpy(ia_usm, ia_host.data(), ia_host.size() * sizeof(intType)));
    dependencies.push_back(
        main_queue.memcpy(ja_usm, ja_host.data(), ja_host.size() * sizeof(intType)));
    dependencies.push_back(main_queue.memcpy(a_usm, a_host.data(), a_host.size() * sizeof(fpType)));
    dependencies.push_back(main_queue.memcpy(x_usm, x_host.data(), x_host.size() * sizeof(fpType)));
    dependencies.push_back(main_queue.memcpy(y_usm, y_host.data(), y_host.size() * sizeof(fpType)));

    fpType* alpha_host_or_usm_ptr = &alpha;
    if (test_scalar_on_device) {
        dependencies.push_back(main_queue.memcpy(alpha_usm_uptr.get(), &alpha, sizeof(fpType)));
        alpha_host_or_usm_ptr = alpha_usm_uptr.get();
    }

    sycl::event ev_copy, ev_spsv;
    oneapi::mkl::sparse::matrix_handle_t A_handle = nullptr;
    oneapi::mkl::sparse::dense_vector_handle_t x_handle = nullptr;
    oneapi::mkl::sparse::dense_vector_handle_t y_handle = nullptr;
    oneapi::mkl::sparse::spsv_descr_t descr = nullptr;
    std::unique_ptr<std::uint8_t, UsmDeleter> workspace_usm(nullptr, UsmDeleter(main_queue));
    try {
        init_sparse_matrix(main_queue, format, &A_handle, m, m, nnz, index, ia_usm, ja_usm, a_usm);
        for (auto property : matrix_properties) {
            CALL_RT_OR_CT(oneapi::mkl::sparse::set_matrix_property, main_queue, A_handle, property);
        }
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_dense_vector, main_queue, &x_handle, m, x_usm);
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_dense_vector, main_queue, &y_handle, m, y_usm);

        CALL_RT_OR_CT(oneapi::mkl::sparse::init_spsv_descr, main_queue, &descr);

        std::size_t workspace_size = 0;
        CALL_RT_OR_CT(oneapi::mkl::sparse::spsv_buffer_size, main_queue, transpose_val,
                      alpha_host_or_usm_ptr, A_view, A_handle, x_handle, y_handle, alg, descr,
                      workspace_size);
        workspace_usm = malloc_device_uptr<std::uint8_t>(main_queue, workspace_size);

        sycl::event ev_opt;
        CALL_RT_OR_CT(ev_opt = oneapi::mkl::sparse::spsv_optimize, main_queue, transpose_val,
                      alpha_host_or_usm_ptr, A_view, A_handle, x_handle, y_handle, alg, descr,
                      workspace_usm.get(), dependencies);

        CALL_RT_OR_CT(ev_spsv = oneapi::mkl::sparse::spsv, main_queue, transpose_val,
                      alpha_host_or_usm_ptr, A_view, A_handle, x_handle, y_handle, alg, descr,
                      { ev_opt });

        if (reset_data) {
            intType reset_nnz = generate_random_matrix<fpType, intType>(
                format, m, m, density_A_matrix, indexing, ia_host, ja_host, a_host, is_symmetric,
                require_diagonal);
            shuffle_sparse_matrix_if_needed(format, matrix_properties, indexing, ia_host.data(),
                                            ja_host.data(), a_host.data(), reset_nnz, mu);
            ev_spsv.wait_and_throw();
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
                                                     ia_host.size() * sizeof(intType), ev_spsv));
            dependencies.push_back(main_queue.memcpy(ja_usm, ja_host.data(),
                                                     ja_host.size() * sizeof(intType), ev_spsv));
            dependencies.push_back(
                main_queue.memcpy(a_usm, a_host.data(), a_host.size() * sizeof(fpType), ev_spsv));
            dependencies.push_back(
                main_queue.memcpy(y_usm, y_host.data(), y_host.size() * sizeof(fpType), ev_spsv));
            set_matrix_data(main_queue, format, A_handle, m, m, nnz, index, ia_usm, ja_usm, a_usm);

            std::size_t workspace_size_2 = 0;
            CALL_RT_OR_CT(oneapi::mkl::sparse::spsv_buffer_size, main_queue, transpose_val,
                          alpha_host_or_usm_ptr, A_view, A_handle, x_handle, y_handle, alg, descr,
                          workspace_size_2);
            if (workspace_size_2 > workspace_size) {
                workspace_usm = malloc_device_uptr<std::uint8_t>(main_queue, workspace_size_2);
            }

            CALL_RT_OR_CT(ev_opt = oneapi::mkl::sparse::spsv_optimize, main_queue, transpose_val,
                          alpha_host_or_usm_ptr, A_view, A_handle, x_handle, y_handle, alg, descr,
                          workspace_usm.get(), dependencies);

            CALL_RT_OR_CT(ev_spsv = oneapi::mkl::sparse::spsv, main_queue, transpose_val,
                          alpha_host_or_usm_ptr, A_view, A_handle, x_handle, y_handle, alg, descr,
                          { ev_opt });
        }

        ev_copy = main_queue.memcpy(y_host.data(), y_usm, y_host.size() * sizeof(fpType), ev_spsv);
    }
    catch (const sycl::exception& e) {
        std::cout << "Caught synchronous SYCL exception during sparse SPSV:\n"
                  << e.what() << std::endl;
        print_error_code(e);
        return 0;
    }
    catch (const oneapi::mkl::unimplemented& e) {
        wait_and_free_handles(main_queue, A_handle, x_handle, y_handle);
        if (descr) {
            sycl::event ev_release_descr;
            CALL_RT_OR_CT(ev_release_descr = oneapi::mkl::sparse::release_spsv_descr, main_queue,
                          descr);
            ev_release_descr.wait();
        }
        return test_skipped;
    }
    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of sparse SPSV:\n" << error.what() << std::endl;
        return 0;
    }
    sycl::event ev_release_descr;
    CALL_RT_OR_CT(ev_release_descr = oneapi::mkl::sparse::release_spsv_descr, main_queue, descr,
                  { ev_spsv });
    ev_release_descr.wait_and_throw();
    free_handles(main_queue, { ev_spsv }, A_handle, x_handle, y_handle);

    // Compute reference.
    prepare_reference_spsv_data(format, ia_host.data(), ja_host.data(), a_host.data(), m, nnz,
                                indexing, transpose_val, x_host.data(), alpha, A_view,
                                y_ref_host.data());

    // Compare the results of reference implementation and DPC++ implementation.
    // Increase default relative error margin for tests that lead to large numeric values.
    ev_copy.wait_and_throw();
    bool valid = check_equal_vector(y_host, y_ref_host);

    return static_cast<int>(valid);
}

class SparseSpsvUsmTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(SparseSpsvUsmTests, RealSinglePrecision) {
    using fpType = float;
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::nontrans, num_passed, num_skipped);
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::trans, num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpsvUsmTests, RealDoublePrecision) {
    using fpType = double;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::nontrans, num_passed, num_skipped);
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::trans, num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpsvUsmTests, ComplexSinglePrecision) {
    using fpType = std::complex<float>;
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::nontrans, num_passed, num_skipped);
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::trans, num_passed, num_skipped);
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::conjtrans, num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

TEST_P(SparseSpsvUsmTests, ComplexDoublePrecision) {
    using fpType = std::complex<double>;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    int num_passed = 0, num_skipped = 0;
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::nontrans, num_passed, num_skipped);
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::trans, num_passed, num_skipped);
    test_helper<fpType>(test_spsv<fpType, int32_t>, test_spsv<fpType, std::int64_t>, GetParam(),
                        oneapi::mkl::transpose::conjtrans, num_passed, num_skipped);
    if (num_skipped > 0) {
        // Mark that some tests were skipped
        GTEST_SKIP() << "Passed: " << num_passed << ", Skipped: " << num_skipped
                     << " configurations." << std::endl;
    }
}

INSTANTIATE_TEST_SUITE_P(SparseSpsvUsmTestSuite, SparseSpsvUsmTests, testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
