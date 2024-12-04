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
#include "oneapi/math/detail/config.hpp"
#include "oneapi/math.hpp"
#include "onemath_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device*> devices;

namespace {

template <typename Ta, typename Tb, typename Tc, typename Ts>
int test(device* dev, oneapi::math::layout layout, int64_t batch_size) {
    // Prepare data.
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    oneapi::math::transpose transa, transb;
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
            transa = oneapi::math::transpose::conjtrans;
        else
            transa = (oneapi::math::transpose)tmp;
        tmp = std::rand() % 3;
        if (tmp == 2)
            transb = oneapi::math::transpose::conjtrans;
        else
            transb = (oneapi::math::transpose)tmp;
    }
    else {
        transa = (oneapi::math::transpose)(std::rand() % 2);
        transb = (oneapi::math::transpose)(std::rand() % 2);
    }

    int64_t stride_a, stride_b, stride_c;

    switch (layout) {
        case oneapi::math::layout::col_major:
            stride_a = (transa == oneapi::math::transpose::nontrans) ? lda * k : lda * m;
            stride_b = (transb == oneapi::math::transpose::nontrans) ? ldb * n : ldb * k;
            stride_c = ldc * n;
            break;
        case oneapi::math::layout::row_major:
            stride_a = (transa == oneapi::math::transpose::nontrans) ? lda * m : lda * k;
            stride_b = (transb == oneapi::math::transpose::nontrans) ? ldb * k : ldb * n;
            stride_c = ldc * m;
            break;
        default: break;
    }

    vector<Ta, allocator_helper<Ta, 64>> A(stride_a * batch_size);
    vector<Ta, allocator_helper<Tb, 64>> B(stride_b * batch_size);
    vector<Tc, allocator_helper<Tc, 64>> C(stride_c * batch_size),
        C_cast_ref(stride_c * batch_size);
    vector<Ts, allocator_helper<Ts, 64>> A_ref(stride_a * batch_size), B_ref(stride_b * batch_size),
        C_ref(stride_c * batch_size);

    for (i = 0; i < batch_size; i++) {
        rand_matrix(A.data() + stride_a * i, layout, transa, m, k, lda);
        rand_matrix(B.data() + stride_b * i, layout, transb, k, n, ldb);
        rand_matrix(C.data() + stride_c * i, layout, oneapi::math::transpose::nontrans, m, n, ldc);
    }

    for (size_t i = 0; i < A.size(); ++i) {
        A_ref[i] = A[i];
    }
    for (size_t i = 0; i < B.size(); ++i) {
        B_ref[i] = B[i];
    }
    for (size_t i = 0; i < C.size(); ++i) {
        C_ref[i] = C[i];
    }

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

    buffer<Ta, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<Tb, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<Tc, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::gemm_batch(
                    main_queue, transa, transb, m, n, k, alpha, A_buffer, lda, stride_a, B_buffer,
                    ldb, stride_b, beta, C_buffer, ldc, stride_c, batch_size);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::gemm_batch(
                    main_queue, transa, transb, m, n, k, alpha, A_buffer, lda, stride_a, B_buffer,
                    ldb, stride_b, beta, C_buffer, ldc, stride_c, batch_size);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::gemm_batch,
                                        transa, transb, m, n, k, alpha, A_buffer, lda, stride_a,
                                        B_buffer, ldb, stride_b, beta, C_buffer, ldc, stride_c,
                                        batch_size);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gemm_batch,
                                        transa, transb, m, n, k, alpha, A_buffer, lda, stride_a,
                                        B_buffer, ldb, stride_b, beta, C_buffer, ldc, stride_c,
                                        batch_size);
                break;
            default: break;
        }
#endif
        main_queue.wait_and_throw();
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
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
    auto C_accessor = C_buffer.get_host_access(read_only);
    bool good = check_almost_equal_matrix(C_accessor, C_cast_ref, oneapi::math::layout::col_major,
                                          stride_c * batch_size, 1, stride_c * batch_size,
                                          error_mag, std::cout);

    return (int)good;
}

class GemmBatchStrideTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(GemmBatchStrideTests, RealHalfPrecision) {
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half, sycl::half, sycl::half>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideTests, HalfHalfFloatPrecision) {
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half, float, float>(std::get<0>(GetParam()),
                                                                  std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideTests, Int8Int8SinglePrecision) {
    EXPECT_TRUEORSKIP((test<std::int8_t, std::int8_t, float, float>(std::get<0>(GetParam()),
                                                                    std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideTests, Int8Int8Int32Precision) {
    EXPECT_TRUEORSKIP((test<std::int8_t, std::int8_t, std::int32_t, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<float, float, float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP((
        test<double, double, double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, std::complex<float>, std::complex<float>, std::complex<float>>(
            std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

TEST_P(GemmBatchStrideTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, std::complex<double>, std::complex<double>,
              std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5)));
}

INSTANTIATE_TEST_SUITE_P(GemmBatchStrideTestSuite, GemmBatchStrideTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
