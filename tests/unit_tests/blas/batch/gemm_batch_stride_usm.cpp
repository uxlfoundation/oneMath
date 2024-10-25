/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "allocator_helper.hpp"
#include "cblas.h"
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device*> devices;

namespace {

template <typename Ta, typename Tb, typename Tc, typename Ts>
int test(device* dev, oneapi::mkl::layout layout, int64_t batch_size) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    oneapi::mkl::transpose transa, transb;
    Ts alpha, beta;

    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    m = 1 + std::rand() % 500;
    n = 1 + std::rand() % 500;
    k = 1 + std::rand() % 500;
    lda = std::max(m, k);
    ldb = std::max(n, k);
    ldc = std::max(m, n);
    alpha = rand_scalar<Ts>();
    beta = rand_scalar<Ts>();
    if ((std::is_same<Ts, std::complex<float>>::value) ||
        (std::is_same<Ts, std::complex<double>>::value)) {
        tmp = std::rand() % 3;
        if (tmp == 2)
            transa = oneapi::mkl::transpose::conjtrans;
        else
            transa = (oneapi::mkl::transpose)tmp;
        tmp = std::rand() % 3;
        if (tmp == 2)
            transb = oneapi::mkl::transpose::conjtrans;
        else
            transb = (oneapi::mkl::transpose)tmp;
    }
    else {
        transa = (oneapi::mkl::transpose)(std::rand() % 2);
        transb = (oneapi::mkl::transpose)(std::rand() % 2);
    }

    int64_t stride_a, stride_b, stride_c;

    switch (layout) {
        case oneapi::mkl::layout::col_major:
            stride_a = (transa == oneapi::mkl::transpose::nontrans) ? lda * k : lda * m;
            stride_b = (transb == oneapi::mkl::transpose::nontrans) ? ldb * n : ldb * k;
            stride_c = ldc * n;
            break;
        case oneapi::mkl::layout::row_major:
            stride_a = (transa == oneapi::mkl::transpose::nontrans) ? lda * m : lda * k;
            stride_b = (transb == oneapi::mkl::transpose::nontrans) ? ldb * k : ldb * n;
            stride_c = ldc * m;
            break;
        default: break;
    }

    auto ua = usm_allocator<Ta, usm::alloc::shared, 64>(cxt, *dev);
    auto ub = usm_allocator<Tb, usm::alloc::shared, 64>(cxt, *dev);
    auto uc = usm_allocator<Tc, usm::alloc::shared, 64>(cxt, *dev);
    auto us = usm_allocator<Ts, usm::alloc::shared, 64>(cxt, *dev);
    vector<Ta, decltype(ua)> A(ua);
    vector<Tb, decltype(ub)> B(ub);
    vector<Tc, decltype(uc)> C(uc), C_cast_ref(uc);
    vector<Ts, decltype(us)> A_ref(us), B_ref(us), C_ref(us);

    A.resize(stride_a * batch_size);
    B.resize(stride_b * batch_size);
    C.resize(stride_c * batch_size);
    A_ref.resize(stride_c * batch_size);
    B_ref.resize(stride_c * batch_size);
    C_ref.resize(stride_c * batch_size);
    C_cast_ref.resize(stride_c * batch_size);

    Ta** a_array = (Ta**)oneapi::mkl::malloc_shared(64, sizeof(Ta*) * batch_size, *dev, cxt);
    Tb** b_array = (Tb**)oneapi::mkl::malloc_shared(64, sizeof(Tb*) * batch_size, *dev, cxt);
    Tc** c_array = (Tc**)oneapi::mkl::malloc_shared(64, sizeof(Tc*) * batch_size, *dev, cxt);
    Ts** c_ref_array = (Ts**)oneapi::mkl::malloc_shared(64, sizeof(Ts*) * batch_size, *dev, cxt);

    if ((a_array == NULL) || (b_array == NULL) || (c_array == NULL) || (c_ref_array == NULL)) {
        std::cout << "Error cannot allocate arrays of pointers\n";
        oneapi::mkl::free_shared(a_array, cxt);
        oneapi::mkl::free_shared(b_array, cxt);
        oneapi::mkl::free_shared(c_array, cxt);
        oneapi::mkl::free_shared(c_ref_array, cxt);
        return false;
    }

    for (i = 0; i < batch_size; i++) {
        a_array[i] = &A[i * stride_a];
        b_array[i] = &B[i * stride_b];
        c_array[i] = &C[i * stride_c];
        c_ref_array[i] = &C_ref[i * stride_c];
    }

    rand_matrix(A, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_a * batch_size, 1, stride_a * batch_size);
    rand_matrix(B, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_b * batch_size, 1, stride_b * batch_size);
    rand_matrix(C, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_c * batch_size, 1, stride_c * batch_size);
    copy_matrix(A, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_a * batch_size, 1, stride_a * batch_size, A_ref);
    copy_matrix(B, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_b * batch_size, 1, stride_b * batch_size, B_ref);
    copy_matrix(C, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_c * batch_size, 1, stride_c * batch_size, C_ref);

    // Call reference GEMM_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<Ts>::type;
    int m_ref = (int)m;
    int n_ref = (int)n;
    int k_ref = (int)k;
    int lda_ref = (int)lda;
    int ldb_ref = (int)ldb;
    int ldc_ref = (int)ldc;
    int batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        ::gemm(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa),
               convert_to_cblas_trans(transb), (const int*)&m_ref, (const int*)&n_ref,
               (const int*)&k_ref, (const fp_ref*)&alpha,
               (const fp_ref*)(A_ref.data() + stride_a * i), (const int*)&lda_ref,
               (const fp_ref*)(B_ref.data() + stride_b * i), (const int*)&ldb_ref,
               (const fp_ref*)&beta, (fp_ref*)(C_ref.data() + stride_c * i), (const int*)&ldc_ref);
    }

    // Call DPC++ GEMM_BATCH_STRIDE.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                done = oneapi::mkl::blas::column_major::gemm_batch(
                    main_queue, transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0], ldb,
                    stride_b, beta, &C[0], ldc, stride_c, batch_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::gemm_batch(
                    main_queue, transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0], ldb,
                    stride_b, beta, &C[0], ldc, stride_c, batch_size, dependencies);
                break;
            default: break;
        }
        done.wait_and_throw();
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::gemm_batch,
                                        transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0],
                                        ldb, stride_b, beta, &C[0], ldc, stride_c, batch_size,
                                        dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::gemm_batch,
                                        transa, transb, m, n, k, alpha, &A[0], lda, stride_a, &B[0],
                                        ldb, stride_b, beta, &C[0], ldc, stride_c, batch_size,
                                        dependencies);
                break;
            default: break;
        }
        main_queue.wait_and_throw();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        oneapi::mkl::free_shared(a_array, cxt);
        oneapi::mkl::free_shared(b_array, cxt);
        oneapi::mkl::free_shared(c_array, cxt);
        oneapi::mkl::free_shared(c_ref_array, cxt);
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    int tol_scalar = 10;
    int error_mag = tol_scalar * k;
    if (std::is_same_v<Tc, int32_t>)
        error_mag = 1;

    for (size_t i = 0; i < C_ref.size(); ++i) {
        C_cast_ref[i] = C_ref[i];
    }
    bool good = check_almost_equal_matrix(C, C_cast_ref, oneapi::mkl::layout::col_major,
                                          stride_c * batch_size, 1, stride_c * batch_size,
                                          error_mag, std::cout);

    oneapi::mkl::free_shared(a_array, cxt);
    oneapi::mkl::free_shared(b_array, cxt);
    oneapi::mkl::free_shared(c_array, cxt);
    oneapi::mkl::free_shared(c_ref_array, cxt);

    return (int)good;
}

class GemmBatchStrideUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(GemmBatchStrideUsmTests, RealHalfPrecision) {
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half, sycl::half, sycl::half>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, HalfHalfFloatPrecision) {
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half, float, float>(std::get<0>(GetParam()),
                                                                  std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, Int8Int8SinglePrecision) {
    EXPECT_TRUEORSKIP((test<std::int8_t, std::int8_t, float, float>(std::get<0>(GetParam()),
                                                                    std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, Int8Int8Int32Precision) {
    EXPECT_TRUEORSKIP((test<std::int8_t, std::int8_t, std::int32_t, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<float, float, float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP((
        test<double, double, double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>>(
            std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, std::complex<double>, std::complex<double>,
              std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

INSTANTIATE_TEST_SUITE_P(GemmBatchStrideUsmTestSuite, GemmBatchStrideUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
