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

template <typename fp>
int test(device* dev, oneapi::mkl::layout layout, int64_t incx, int64_t incy, int64_t batch_size) {
    // Prepare data.
    int64_t m, n;
    int64_t lda;
    oneapi::mkl::transpose transa;
    fp alpha, beta;
    int64_t i, tmp;

    batch_size = 15;
    m = 25;
    n = 30;
    lda = 42;
    alpha = rand_scalar<fp>();
    beta = rand_scalar<fp>();

    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        transa = (oneapi::mkl::transpose)(std::rand() % 2);
    }
    else {
        tmp = std::rand() % 3;
        if (tmp == 2)
            transa = oneapi::mkl::transpose::conjtrans;
        else
            transa = (oneapi::mkl::transpose)tmp;
    }

    int x_len = outer_dimension(transa, m, n);
    int y_len = inner_dimension(transa, m, n);

    int64_t stride_x, stride_y, stride_a;
    stride_x = x_len * std::abs(incx);
    stride_y = y_len * std::abs(incy);
    stride_a = lda * std::max(m, n);

    vector<fp, allocator_helper<fp, 64>> x(stride_x * batch_size), A(stride_a * batch_size);
    vector<fp, allocator_helper<fp, 64>> y(stride_y * batch_size), y_ref(stride_y * batch_size);

    for (i = 0; i < batch_size; i++) {
        rand_vector(x.data() + stride_x * i, x_len, incx);
        rand_vector(y.data() + stride_y * i, y_len, incy);
        rand_matrix(A.data() + stride_a * i, layout, oneapi::mkl::transpose::nontrans, m, n, lda);
    }

    y_ref = y;

    // Call reference GEMV_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<fp>::type;
    int m_ref = (int)m;
    int n_ref = (int)n;
    int incx_ref = (int)incx;
    int incy_ref = (int)incy;
    int lda_ref = (int)lda;
    int batch_size_ref = (int)batch_size;

    for (i = 0; i < batch_size_ref; i++) {
        ::gemv(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa), (const int*)&m_ref,
               (const int*)&n_ref, (const fp_ref*)&alpha, (const fp_ref*)(A.data() + stride_a * i),
               (const int*)&lda_ref, (const fp_ref*)(x.data() + stride_x * i),
               (const int*)&incx_ref, (const fp_ref*)&beta, (fp_ref*)(y_ref.data() + stride_y * i),
               (const int*)&incy_ref);
    }

    // Call DPC++ GEMV_BATCH_STRIDE.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMV_BATCH_STRIDE:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer(x.data(), range<1>(x.size()));
    buffer<fp, 1> y_buffer(y.data(), range<1>(y.size()));
    buffer<fp, 1> A_buffer(A.data(), range<1>(A.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                oneapi::mkl::blas::column_major::gemv_batch(
                    main_queue, transa, m, n, alpha, A_buffer, lda, stride_a, x_buffer, incx,
                    stride_x, beta, y_buffer, incy, stride_y, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::gemv_batch(
                    main_queue, transa, m, n, alpha, A_buffer, lda, stride_a, x_buffer, incx,
                    stride_x, beta, y_buffer, incy, stride_y, batch_size);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::gemv_batch,
                                        transa, m, n, alpha, A_buffer, lda, stride_a, x_buffer,
                                        incx, stride_x, beta, y_buffer, incy, stride_y, batch_size);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::gemv_batch,
                                        transa, m, n, alpha, A_buffer, lda, stride_a, x_buffer,
                                        incx, stride_x, beta, y_buffer, incy, stride_y, batch_size);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMV_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMV_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto y_accessor = y_buffer.get_host_access(read_only);
    bool good = true;
    for (i = 0; i < batch_size; i++) {
        good = good && check_equal_vector(y_accessor.get_pointer() + i * stride_y,
                                          y_ref.data() + i * stride_y, y_len, incy,
                                          std::max<int>(m, n), std::cout);
    }
    return (int)good;
}

class GemvBatchStrideTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(GemvBatchStrideTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, 5));
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, 5));
}

TEST_P(GemvBatchStrideTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, 5));
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, 5));
}

TEST_P(GemvBatchStrideTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, 5);
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, 5);
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, 5));
}

TEST_P(GemvBatchStrideTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 2, 3, 5);
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), -2, -3, 5);
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1, 1, 5));
}

INSTANTIATE_TEST_SUITE_P(GemvBatchStrideTestSuite, GemvBatchStrideTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
