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

#ifndef _TEST_SPMV_HPP__
#define _TEST_SPMV_HPP__

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl.hpp"
#include "oneapi/mkl/detail/config.hpp"

#include "common_sparse_reference.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

/**
 * Helper function to run tests in different configuration.
 *
 * @tparam fpType Complex or scalar, single or double precision type
 * @tparam testFunctorI32 Test functor for fpType and int32
 * @tparam testFunctorI64 Test functor for fpType and int64
 * @param dev Device to test
 * @param format Sparse matrix format to use
 * @param non_default_algorithms Algorithms compatible with the given format, other than default_alg
 * @param transpose_val Transpose value for the input matrix
 * @param num_passed Increase the number of configurations passed
 * @param num_skipped Increase the number of configurations skipped
 *
 * The test functions will use different sizes if the configuration implies a symmetric matrix.
 */
template <typename fpType, typename testFunctorI32, typename testFunctorI64>
void test_helper_with_format_with_transpose(
    testFunctorI32 test_functor_i32, testFunctorI64 test_functor_i64, sycl::device *dev,
    sparse_matrix_format_t format,
    const std::vector<oneapi::mkl::sparse::spmv_alg> &non_default_algorithms,
    oneapi::mkl::transpose transpose_val, int &num_passed, int &num_skipped) {
    sycl::property_list queue_properties;
    double density_A_matrix = 0.8;
    fpType fp_zero = set_fp_value<fpType>()(0.f, 0.f);
    fpType fp_one = set_fp_value<fpType>()(1.f, 0.f);
    int nrows_A = 4, ncols_A = 6;
    oneapi::mkl::index_base index_zero = oneapi::mkl::index_base::zero;
    oneapi::mkl::sparse::spmv_alg default_alg = oneapi::mkl::sparse::spmv_alg::default_alg;
    oneapi::mkl::sparse::matrix_view default_A_view;
    bool no_reset_data = false;
    bool no_scalars_on_device = false;

    // Queue is only used to get which matrix_property should be used for the tests.
    sycl::queue properties_queue(*dev);
    auto default_properties = get_default_matrix_properties(properties_queue, format);

    // Basic test
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, default_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Reset data
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, default_A_view,
                         default_properties, true, no_scalars_on_device),
        num_passed, num_skipped);
    // Test alpha and beta on the device
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, default_A_view,
                         default_properties, no_reset_data, true),
        num_passed, num_skipped);
    // Test index_base 1
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         oneapi::mkl::index_base::one, transpose_val, fp_one, fp_zero, default_alg,
                         default_A_view, default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test non-default alpha
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, set_fp_value<fpType>()(2.f, 1.5f), fp_zero,
                         default_alg, default_A_view, default_properties, no_reset_data,
                         no_scalars_on_device),
        num_passed, num_skipped);
    // Test non-default beta
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, set_fp_value<fpType>()(3.2f, 1.f),
                         default_alg, default_A_view, default_properties, no_reset_data,
                         no_scalars_on_device),
        num_passed, num_skipped);
    // Test 0 alpha
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_zero, fp_one, default_alg, default_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test 0 alpha and beta
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_zero, fp_zero, default_alg, default_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test int64 indices
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i64(dev, queue_properties, format, 27L, 13L, density_A_matrix, index_zero,
                         transpose_val, fp_one, fp_zero, default_alg, default_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Lower triangular
    oneapi::mkl::sparse::matrix_view triangular_A_view(
        oneapi::mkl::sparse::matrix_descr::triangular);
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, triangular_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Upper triangular
    triangular_A_view.uplo_view = oneapi::mkl::uplo::upper;
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, triangular_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Lower triangular unit diagonal
    oneapi::mkl::sparse::matrix_view triangular_unit_A_view(
        oneapi::mkl::sparse::matrix_descr::triangular);
    triangular_unit_A_view.diag_view = oneapi::mkl::diag::unit;
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg,
                         triangular_unit_A_view, default_properties, no_reset_data,
                         no_scalars_on_device),
        num_passed, num_skipped);
    // Upper triangular unit diagonal
    triangular_A_view.uplo_view = oneapi::mkl::uplo::upper;
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg,
                         triangular_unit_A_view, default_properties, no_reset_data,
                         no_scalars_on_device),
        num_passed, num_skipped);
    // Lower symmetric
    oneapi::mkl::sparse::matrix_view symmetric_view(oneapi::mkl::sparse::matrix_descr::symmetric);
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, symmetric_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Upper symmetric
    symmetric_view.uplo_view = oneapi::mkl::uplo::upper;
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, symmetric_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Lower hermitian
    oneapi::mkl::sparse::matrix_view hermitian_view(oneapi::mkl::sparse::matrix_descr::hermitian);
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, hermitian_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Upper hermitian
    hermitian_view.uplo_view = oneapi::mkl::uplo::upper;
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                         index_zero, transpose_val, fp_one, fp_zero, default_alg, hermitian_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test other algorithms
    for (auto alg : non_default_algorithms) {
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                             index_zero, transpose_val, fp_one, fp_zero, alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
    }
    // Test matrix properties
    for (auto properties : get_all_matrix_properties_combinations(properties_queue, format)) {
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, density_A_matrix,
                             index_zero, transpose_val, fp_one, fp_zero, default_alg,
                             default_A_view, properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
    }
    // In-order queue
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, { property::queue::in_order::in_order() }, format, nrows_A, ncols_A,
                         density_A_matrix, index_zero, transpose_val, fp_one, fp_zero, default_alg,
                         default_A_view, default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
}

/**
 * Helper function to test combination of transpose vals.
 *
 * @tparam fpType Complex or scalar, single or double precision type
 * @tparam testFunctorI32 Test functor for fpType and int32
 * @tparam testFunctorI64 Test functor for fpType and int64
 * @param dev Device to test
 * @param format Sparse matrix format to use
 * @param non_default_algorithms Algorithms compatible with the given format, other than default_alg
 * @param num_passed Increase the number of configurations passed
 * @param num_skipped Increase the number of configurations skipped
 */
template <typename fpType, typename testFunctorI32, typename testFunctorI64>
void test_helper_with_format(
    testFunctorI32 test_functor_i32, testFunctorI64 test_functor_i64, sycl::device *dev,
    sparse_matrix_format_t format,
    const std::vector<oneapi::mkl::sparse::spmv_alg> &non_default_algorithms, int &num_passed,
    int &num_skipped) {
    std::vector<oneapi::mkl::transpose> transpose_vals{ oneapi::mkl::transpose::nontrans,
                                                        oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::conjtrans };
    for (auto transpose_A : transpose_vals) {
        test_helper_with_format_with_transpose<fpType>(test_functor_i32, test_functor_i64, dev,
                                                       format, non_default_algorithms, transpose_A,
                                                       num_passed, num_skipped);
    }
}

/**
 * Helper function to test multiple sparse matrix format and choose valid algorithms.
 *
 * @tparam fpType Complex or scalar, single or double precision type
 * @tparam testFunctorI32 Test functor for fpType and int32
 * @tparam testFunctorI64 Test functor for fpType and int64
 * @param dev Device to test
 * @param num_passed Increase the number of configurations passed
 * @param num_skipped Increase the number of configurations skipped
 */
template <typename fpType, typename testFunctorI32, typename testFunctorI64>
void test_helper(testFunctorI32 test_functor_i32, testFunctorI64 test_functor_i64,
                 sycl::device *dev, int &num_passed, int &num_skipped) {
    test_helper_with_format<fpType>(
        test_functor_i32, test_functor_i64, dev, sparse_matrix_format_t::CSR,
        { oneapi::mkl::sparse::spmv_alg::no_optimize_alg, oneapi::mkl::sparse::spmv_alg::csr_alg1,
          oneapi::mkl::sparse::spmv_alg::csr_alg2, oneapi::mkl::sparse::spmv_alg::csr_alg3 },
        num_passed, num_skipped);
    test_helper_with_format<fpType>(
        test_functor_i32, test_functor_i64, dev, sparse_matrix_format_t::COO,
        { oneapi::mkl::sparse::spmv_alg::no_optimize_alg, oneapi::mkl::sparse::spmv_alg::coo_alg1,
          oneapi::mkl::sparse::spmv_alg::coo_alg2 },
        num_passed, num_skipped);
}

/// Compute spmv reference as a dense operation
template <typename fpType, typename intType>
void prepare_reference_spmv_data(sparse_matrix_format_t format, const intType *ia,
                                 const intType *ja, const fpType *a, intType a_nrows,
                                 intType a_ncols, intType a_nnz, intType indexing,
                                 oneapi::mkl::transpose opA, fpType alpha, fpType beta,
                                 const fpType *x, oneapi::mkl::sparse::matrix_view A_view,
                                 fpType *y_ref) {
    std::size_t a_nrows_u = static_cast<std::size_t>(a_nrows);
    std::size_t a_ncols_u = static_cast<std::size_t>(a_ncols);
    auto [opa_nrows, opa_ncols] = swap_if_transposed(opA, a_nrows_u, a_ncols_u);
    const std::size_t nnz = static_cast<std::size_t>(a_nnz);
    auto dense_opa =
        sparse_to_dense(format, ia, ja, a, a_nrows_u, a_ncols_u, nnz, indexing, opA, A_view);

    //
    // do SPMV operation
    //
    //  y_ref <- alpha * op(A) * x + beta * y_ref
    //
    for (std::size_t row = 0; row < opa_nrows; row++) {
        fpType acc = 0;
        for (std::size_t col = 0; col < opa_ncols; col++) {
            acc += dense_opa[row * opa_ncols + col] * x[col];
        }
        y_ref[row] = alpha * acc + beta * y_ref[row];
    }
}

#endif // _TEST_SPMV_HPP__
