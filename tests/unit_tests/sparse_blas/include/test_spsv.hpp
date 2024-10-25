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

#ifndef _TEST_SPSV_HPP__
#define _TEST_SPSV_HPP__

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
 * @param transpose_val Transpose value for the input matrix
 * @param num_passed Increase the number of configurations passed
 * @param num_skipped Increase the number of configurations skipped
 */
template <typename fpType, typename testFunctorI32, typename testFunctorI64>
void test_helper_with_format(testFunctorI32 test_functor_i32, testFunctorI64 test_functor_i64,
                             sycl::device* dev, sparse_matrix_format_t format,
                             oneapi::mkl::transpose transpose_val, int& num_passed,
                             int& num_skipped) {
    sycl::property_list queue_properties;
    double density_A_matrix = 0.144;
    fpType alpha = set_fp_value<fpType>()(1.f, 0.f);
    int m = 277;
    oneapi::mkl::index_base index_zero = oneapi::mkl::index_base::zero;
    oneapi::mkl::sparse::spsv_alg default_alg = oneapi::mkl::sparse::spsv_alg::default_alg;
    oneapi::mkl::sparse::spsv_alg no_optimize_alg = oneapi::mkl::sparse::spsv_alg::no_optimize_alg;
    oneapi::mkl::sparse::matrix_view default_A_view(oneapi::mkl::sparse::matrix_descr::triangular);
    oneapi::mkl::sparse::matrix_view upper_A_view(oneapi::mkl::sparse::matrix_descr::triangular);
    upper_A_view.uplo_view = oneapi::mkl::uplo::upper;
    bool no_reset_data = false;
    bool no_scalars_on_device = false;

    // Queue is only used to get which matrix_property should be used for the tests.
    sycl::queue properties_queue(*dev);
    auto default_properties = get_default_matrix_properties(properties_queue, format);

    // Basic test
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, alpha, default_alg, default_A_view, default_properties,
                         no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Reset data
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, alpha, default_alg, default_A_view, default_properties,
                         true, no_scalars_on_device),
        num_passed, num_skipped);
    // Test alpha on the device
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, alpha, default_alg, default_A_view, default_properties,
                         no_reset_data, true),
        num_passed, num_skipped);
    // Test index_base 1
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix,
                         oneapi::mkl::index_base::one, transpose_val, alpha, default_alg,
                         default_A_view, default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test upper triangular matrix
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, alpha, default_alg, upper_A_view, default_properties,
                         no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test lower triangular unit diagonal matrix
    oneapi::mkl::sparse::matrix_view triangular_unit_A_view(
        oneapi::mkl::sparse::matrix_descr::triangular);
    triangular_unit_A_view.diag_view = oneapi::mkl::diag::unit;
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, alpha, default_alg, triangular_unit_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test upper triangular unit diagonal matrix
    triangular_unit_A_view.uplo_view = oneapi::mkl::uplo::upper;
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, alpha, default_alg, triangular_unit_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test non-default alpha
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, set_fp_value<fpType>()(2.f, 1.5f), default_alg,
                         default_A_view, default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test int64 indices
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i64(dev, queue_properties, format, 15L, density_A_matrix, index_zero,
                         transpose_val, alpha, default_alg, default_A_view, default_properties,
                         no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test lower no_optimize_alg
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, alpha, no_optimize_alg, default_A_view, default_properties,
                         no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test upper no_optimize_alg
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                         transpose_val, alpha, no_optimize_alg, upper_A_view, default_properties,
                         no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
    // Test matrix properties
    for (auto properties : get_all_matrix_properties_combinations(properties_queue, format)) {
        // Basic test with matrix properties
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                             transpose_val, alpha, default_alg, default_A_view, properties,
                             no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
        // Test lower no_optimize_alg with matrix properties
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, m, density_A_matrix, index_zero,
                             transpose_val, alpha, no_optimize_alg, default_A_view, properties,
                             no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
    }
    // In-order queue
    EXPECT_TRUE_OR_FUTURE_SKIP(
        test_functor_i32(dev, { sycl::property::queue::in_order{} }, format, m, density_A_matrix,
                         index_zero, transpose_val, alpha, default_alg, default_A_view,
                         default_properties, no_reset_data, no_scalars_on_device),
        num_passed, num_skipped);
}

/**
 * Helper function to test multiple sparse matrix format.
 *
 * @tparam fpType Complex or scalar, single or double precision type
 * @tparam testFunctorI32 Test functor for fpType and int32
 * @tparam testFunctorI64 Test functor for fpType and int64
 * @param dev Device to test
 * @param transpose_val Transpose value for the input matrix
 * @param num_passed Increase the number of configurations passed
 * @param num_skipped Increase the number of configurations skipped
 */
template <typename fpType, typename testFunctorI32, typename testFunctorI64>
void test_helper(testFunctorI32 test_functor_i32, testFunctorI64 test_functor_i64,
                 sycl::device* dev, oneapi::mkl::transpose transpose_val, int& num_passed,
                 int& num_skipped) {
    test_helper_with_format<fpType>(test_functor_i32, test_functor_i64, dev,
                                    sparse_matrix_format_t::CSR, transpose_val, num_passed,
                                    num_skipped);
    test_helper_with_format<fpType>(test_functor_i32, test_functor_i64, dev,
                                    sparse_matrix_format_t::COO, transpose_val, num_passed,
                                    num_skipped);
}

/// Compute spsv reference as a dense operation
template <typename fpType, typename intType>
void prepare_reference_spsv_data(sparse_matrix_format_t format, const intType* ia,
                                 const intType* ja, const fpType* a, intType m, intType nnz,
                                 intType indexing, oneapi::mkl::transpose opA, const fpType* x,
                                 fpType alpha, oneapi::mkl::sparse::matrix_view A_view,
                                 fpType* y_ref) {
    std::size_t mu = static_cast<std::size_t>(m);
    auto dense_opa = sparse_to_dense(format, ia, ja, a, mu, mu, static_cast<std::size_t>(nnz),
                                     indexing, opA, A_view);

    //
    // do SPSV operation
    //
    //  y_ref <- op(A)^-1 * x
    //
    // Compute each element of the reference one after the other starting from 0 (resp. the end) for a lower (resp. upper) triangular matrix.
    // A matrix is considered lowered if it is lower and not transposed or upper and transposed.
    const bool is_lower =
        (A_view.uplo_view == oneapi::mkl::uplo::lower) == (opA == oneapi::mkl::transpose::nontrans);
    for (std::size_t row = 0; row < mu; row++) {
        std::size_t uplo_row = is_lower ? row : (mu - 1 - row);
        fpType rhs = alpha * x[uplo_row];
        for (std::size_t col = 0; col < row; col++) {
            std::size_t uplo_col = is_lower ? col : (mu - 1 - col);
            rhs -= dense_opa[uplo_row * mu + uplo_col] * y_ref[uplo_col];
        }
        y_ref[uplo_row] = rhs / dense_opa[uplo_row * mu + uplo_row];
    }
}

#endif // _TEST_SPSV_HPP__
