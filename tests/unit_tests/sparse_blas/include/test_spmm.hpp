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

#ifndef _TEST_SPMM_HPP__
#define _TEST_SPMM_HPP__

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
 * @param transpose_A Transpose value for the A matrix
 * @param transpose_B Transpose value for the B matrix
 * @param num_passed Increase the number of configurations passed
 * @param num_skipped Increase the number of configurations skipped
 *
 * The test functions will use different sizes and leading dimensions if the configuration implies a symmetric matrix.
 */
template <typename fpType, typename testFunctorI32, typename testFunctorI64>
void test_helper_with_format_with_transpose(
    testFunctorI32 test_functor_i32, testFunctorI64 test_functor_i64, sycl::device *dev,
    sparse_matrix_format_t format,
    const std::vector<oneapi::mkl::sparse::spmm_alg> &non_default_algorithms,
    oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B, int &num_passed,
    int &num_skipped) {
    sycl::property_list queue_properties;
    double density_A_matrix = 0.8;
    fpType fp_zero = set_fp_value<fpType>()(0.f, 0.f);
    fpType fp_one = set_fp_value<fpType>()(1.f, 0.f);
    oneapi::mkl::index_base index_zero = oneapi::mkl::index_base::zero;
    oneapi::mkl::layout col_major = oneapi::mkl::layout::col_major;
    oneapi::mkl::sparse::spmm_alg default_alg = oneapi::mkl::sparse::spmm_alg::default_alg;
    oneapi::mkl::sparse::matrix_view default_A_view;
    bool no_reset_data = false;
    bool no_scalars_on_device = false;

    // Queue is only used to get which matrix_property should be used for the tests.
    sycl::queue properties_queue(*dev);
    auto default_properties = get_default_matrix_properties(properties_queue, format);

    {
        int m = 4, k = 6, n = 5;
        int nrows_A = (transpose_A != oneapi::mkl::transpose::nontrans) ? k : m;
        int ncols_A = (transpose_A != oneapi::mkl::transpose::nontrans) ? m : k;
        int nrows_B = (transpose_B != oneapi::mkl::transpose::nontrans) ? n : k;
        int ncols_B = (transpose_B != oneapi::mkl::transpose::nontrans) ? k : n;
        int nrows_C = m;
        int ncols_C = n;
        int ldb = nrows_B;
        int ldc = nrows_C;

        // Basic test
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_one, fp_zero, ldb, ldc, default_alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
        // Reset data
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_one, fp_zero, ldb, ldc, default_alg, default_A_view,
                             default_properties, true, no_scalars_on_device),
            num_passed, num_skipped);
        // Test alpha and beta on the device
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_one, fp_zero, ldb, ldc, default_alg, default_A_view,
                             default_properties, no_reset_data, true),
            num_passed, num_skipped);
        // Test index_base 1
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, oneapi::mkl::index_base::one, col_major, transpose_A,
                             transpose_B, fp_one, fp_zero, ldb, ldc, default_alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
        // Test non-default alpha
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             set_fp_value<fpType>()(2.f, 1.5f), fp_zero, ldb, ldc, default_alg,
                             default_A_view, default_properties, no_reset_data,
                             no_scalars_on_device),
            num_passed, num_skipped);
        // Test non-default beta
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_one, set_fp_value<fpType>()(3.2f, 1.f), ldb, ldc, default_alg,
                             default_A_view, default_properties, no_reset_data,
                             no_scalars_on_device),
            num_passed, num_skipped);
        // Test 0 alpha
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_zero, fp_one, ldb, ldc, default_alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
        // Test 0 alpha and beta
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_zero, fp_zero, ldb, ldc, default_alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
        // Test non-default ldb
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_one, fp_zero, ldb + 5, ldc, default_alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
        // Test non-default ldc
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_one, fp_zero, ldb, ldc + 6, default_alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
        // Test row major layout
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, oneapi::mkl::layout::row_major,
                             transpose_A, transpose_B, fp_one, fp_zero, ncols_B, ncols_C,
                             default_alg, default_A_view, default_properties, no_reset_data,
                             no_scalars_on_device),
            num_passed, num_skipped);
        // Test int64 indices
        long long_nrows_A = 27, long_ncols_A = 13, long_ncols_C = 6;
        auto [long_ldc, long_ldb] = swap_if_transposed(transpose_A, long_nrows_A, long_ncols_A);
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i64(dev, queue_properties, format, long_nrows_A, long_ncols_A,
                             long_ncols_C, density_A_matrix, index_zero, col_major, transpose_A,
                             transpose_B, fp_one, fp_zero, long_ldb, long_ldc, default_alg,
                             default_A_view, default_properties, no_reset_data,
                             no_scalars_on_device),
            num_passed, num_skipped);
        // Test other algorithms
        for (auto alg : non_default_algorithms) {
            EXPECT_TRUE_OR_FUTURE_SKIP(
                test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                                 density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                                 fp_one, fp_zero, ldb, ldc, alg, default_A_view, default_properties,
                                 no_reset_data, no_scalars_on_device),
                num_passed, num_skipped);
        }
        // Test matrix properties
        for (auto properties : get_all_matrix_properties_combinations(properties_queue, format)) {
            EXPECT_TRUE_OR_FUTURE_SKIP(
                test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                                 density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                                 fp_one, fp_zero, ldb, ldc, default_alg, default_A_view, properties,
                                 no_reset_data, no_scalars_on_device),
                num_passed, num_skipped);
        }
        // In-order queue
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, { property::queue::in_order::in_order() }, format, nrows_A,
                             ncols_A, ncols_C, density_A_matrix, index_zero, col_major, transpose_A,
                             transpose_B, fp_one, fp_zero, ldb, ldc, default_alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
    }
    {
        // Test different sizes
        int m = 6, k = 2, n = 5;
        int nrows_A = (transpose_A != oneapi::mkl::transpose::nontrans) ? k : m;
        int ncols_A = (transpose_A != oneapi::mkl::transpose::nontrans) ? m : k;
        int nrows_B = (transpose_B != oneapi::mkl::transpose::nontrans) ? n : k;
        int nrows_C = m;
        int ncols_C = n;
        int ldb = nrows_B;
        int ldc = nrows_C;
        EXPECT_TRUE_OR_FUTURE_SKIP(
            test_functor_i32(dev, queue_properties, format, nrows_A, ncols_A, ncols_C,
                             density_A_matrix, index_zero, col_major, transpose_A, transpose_B,
                             fp_one, fp_zero, ldb, ldc, default_alg, default_A_view,
                             default_properties, no_reset_data, no_scalars_on_device),
            num_passed, num_skipped);
    }
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
    const std::vector<oneapi::mkl::sparse::spmm_alg> &non_default_algorithms, int &num_passed,
    int &num_skipped) {
    std::vector<oneapi::mkl::transpose> transpose_vals{ oneapi::mkl::transpose::nontrans,
                                                        oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::conjtrans };
    for (auto transpose_A : transpose_vals) {
        for (auto transpose_B : transpose_vals) {
            test_helper_with_format_with_transpose<fpType>(
                test_functor_i32, test_functor_i64, dev, format, non_default_algorithms,
                transpose_A, transpose_B, num_passed, num_skipped);
        }
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
        { oneapi::mkl::sparse::spmm_alg::no_optimize_alg, oneapi::mkl::sparse::spmm_alg::csr_alg1,
          oneapi::mkl::sparse::spmm_alg::csr_alg2, oneapi::mkl::sparse::spmm_alg::csr_alg3 },
        num_passed, num_skipped);
    test_helper_with_format<fpType>(
        test_functor_i32, test_functor_i64, dev, sparse_matrix_format_t::COO,
        { oneapi::mkl::sparse::spmm_alg::no_optimize_alg, oneapi::mkl::sparse::spmm_alg::coo_alg1,
          oneapi::mkl::sparse::spmm_alg::coo_alg2, oneapi::mkl::sparse::spmm_alg::coo_alg3,
          oneapi::mkl::sparse::spmm_alg::coo_alg4 },
        num_passed, num_skipped);
}

/// Compute spmm reference as a dense operation
template <typename fpType, typename intType>
void prepare_reference_spmm_data(sparse_matrix_format_t format, const intType *ia,
                                 const intType *ja, const fpType *a, intType a_nrows,
                                 intType a_ncols, intType c_ncols, intType a_nnz, intType indexing,
                                 oneapi::mkl::layout dense_matrix_layout,
                                 oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                                 fpType alpha, fpType beta, intType ldb, intType ldc,
                                 const fpType *b, oneapi::mkl::sparse::matrix_view A_view,
                                 fpType *c_ref) {
    std::size_t a_nrows_u = static_cast<std::size_t>(a_nrows);
    std::size_t a_ncols_u = static_cast<std::size_t>(a_ncols);
    std::size_t c_ncols_u = static_cast<std::size_t>(c_ncols);
    auto [opa_nrows, opa_ncols] = swap_if_transposed(opA, a_nrows_u, a_ncols_u);
    const std::size_t nnz = static_cast<std::size_t>(a_nnz);
    const std::size_t ldb_u = static_cast<std::size_t>(ldb);
    const std::size_t ldc_u = static_cast<std::size_t>(ldc);
    // dense_opa is always row major
    auto dense_opa =
        sparse_to_dense(format, ia, ja, a, a_nrows_u, a_ncols_u, nnz, indexing, opA, A_view);

    // dense_opb is always row major and not transposed
    auto dense_opb = extract_dense_matrix(b, opa_ncols, c_ncols_u, ldb_u, opB, dense_matrix_layout);

    // Return the linear index to access a dense matrix from
    auto dense_linear_idx = [=](std::size_t row, std::size_t col, std::size_t ld) {
        return (dense_matrix_layout == oneapi::mkl::layout::row_major) ? row * ld + col
                                                                       : col * ld + row;
    };

    //
    // do SPMM operation
    //
    //  C <- alpha * opA(A) * opB(B) + beta * C
    //
    for (std::size_t row = 0; row < opa_nrows; row++) {
        for (std::size_t col = 0; col < c_ncols_u; col++) {
            fpType acc = 0;
            for (std::size_t i = 0; i < opa_ncols; i++) {
                acc += dense_opa[row * opa_ncols + i] * dense_opb[i * c_ncols_u + col];
            }
            fpType &c = c_ref[dense_linear_idx(row, col, ldc_u)];
            c = alpha * acc + beta * c;
        }
    }
}

#endif // _TEST_SPMM_HPP__
