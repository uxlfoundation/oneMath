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

template <typename Ts, typename Ta, typename Tb, typename Tc>
int test(device* dev, oneapi::math::layout layout, oneapi::math::transpose transa,
         oneapi::math::transpose transb, oneapi::math::offset offsetc, int m, int n, int k, int lda,
         int ldb, int ldc, Ts alpha, Ts beta) {
    // Prepare data.
    vector<Ta, allocator_helper<Ta, 64>> A;
    vector<Tb, allocator_helper<Tb, 64>> B;
    vector<Tc, allocator_helper<Tc, 64>> C, C_ref, co;

    Ta ao = rand_scalar<Ta>();
    Tb bo = rand_scalar<Tb>();

    rand_matrix(A, layout, transa, m, k, lda);
    rand_matrix(B, layout, transb, k, n, ldb);
    rand_matrix(C, layout, oneapi::math::transpose::nontrans, m, n, ldc);
    if (offsetc == oneapi::math::offset::fix)
        rand_matrix(co, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, 1, 1,
                    1);
    if (offsetc == oneapi::math::offset::column)
        rand_matrix(co, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, m, 1,
                    m);
    if (offsetc == oneapi::math::offset::row)
        rand_matrix(co, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans, n, 1,
                    n);

    C_ref = C;

    // Call Reference GEMM_BIAS.
    const int m_ref = m, n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using Ts_ref = typename ref_type_info<Ts>::type;
    using Ta_ref = typename ref_type_info<Ta>::type;
    using Tb_ref = typename ref_type_info<Tb>::type;
    using Tc_ref = typename ref_type_info<Tc>::type;

    ::gemm_bias(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa),
                convert_to_cblas_trans(transb), convert_to_cblas_offset(offsetc), &m_ref, &n_ref,
                &k_ref, (Ts_ref*)&alpha, (Ta_ref*)A.data(), &lda_ref, (Ta_ref*)&ao,
                (Tb_ref*)B.data(), &ldb_ref, (Tb_ref*)&bo, (Ts_ref*)&beta, (Tc_ref*)C_ref.data(),
                &ldc_ref, (Tc_ref*)co.data());

    // Call DPC++ GEMM_BIAS.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BIAS:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<Ta, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<Tb, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<Tc, 1> C_buffer(C.data(), range<1>(C.size()));
    buffer<Tc, 1> CO_buffer(co.data(), range<1>(co.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::gemm_bias(
                    main_queue, transa, transb, offsetc, m, n, k, alpha, A_buffer, lda, ao,
                    B_buffer, ldb, bo, beta, C_buffer, ldc, CO_buffer);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::gemm_bias(main_queue, transa, transb, offsetc, m, n,
                                                         k, alpha, A_buffer, lda, ao, B_buffer, ldb,
                                                         bo, beta, C_buffer, ldc, CO_buffer);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::gemm_bias,
                                        transa, transb, offsetc, m, n, k, alpha, A_buffer, lda, ao,
                                        B_buffer, ldb, bo, beta, C_buffer, ldc, CO_buffer);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gemm_bias,
                                        transa, transb, offsetc, m, n, k, alpha, A_buffer, lda, ao,
                                        B_buffer, ldb, bo, beta, C_buffer, ldc, CO_buffer);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BIAS:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM_BIAS:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto C_accessor = C_buffer.get_host_access(read_only);
    bool good = check_equal_matrix(C_accessor, C_ref, layout, m, n, ldc, 10 * k, std::cout);

    return (int)good;
}

class GemmBiasTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(GemmBiasTests, Int8Int8Int32Precision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
}

TEST_P(GemmBiasTests, Int8Uint8Int32Precision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, int8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
}

TEST_P(GemmBiasTests, Uint8Int8Int32Precision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, int8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
}

TEST_P(GemmBiasTests, Uint8Uint8Int32Precision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::fix, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::column, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::nontrans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::nontrans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106,
        alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, uint8_t, uint8_t, int32_t>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::math::transpose::trans,
        oneapi::math::transpose::trans, oneapi::math::offset::row, 79, 83, 91, 103, 105, 106, alpha,
        beta)));
}

INSTANTIATE_TEST_SUITE_P(GemmBiasTestSuite, GemmBiasTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
