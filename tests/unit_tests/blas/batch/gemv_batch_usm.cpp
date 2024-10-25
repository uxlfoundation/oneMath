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
#include "cblas.h"
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include "allocator_helper.hpp"
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
int test(device* dev, oneapi::mkl::layout layout, int64_t group_count) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMV_BATCH:\n"
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
    auto uaint = usm_allocator<int64_t, usm::alloc::shared, 64>(cxt, *dev);
    vector<int64_t, decltype(uaint)> m(uaint), n(uaint), lda(uaint), incx(uaint), incy(uaint),
        group_size(uaint);

    auto uatranspose = usm_allocator<oneapi::mkl::transpose, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::mkl::transpose, decltype(uatranspose)> transa(uatranspose);

    auto uafp = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(uafp)> alpha(uafp), beta(uafp);

    m.resize(group_count);
    n.resize(group_count);
    lda.resize(group_count);
    incx.resize(group_count);
    incy.resize(group_count);
    group_size.resize(group_count);
    transa.resize(group_count);
    alpha.resize(group_count);
    beta.resize(group_count);

    int64_t i, tmp;
    int64_t j, idx = 0;
    int64_t total_batch_count = 0;
    int64_t x_len = 0, y_len = 0;
    int64_t size_a = 0, size_x = 0, size_y = 0;

    for (i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 20;
        m[i] = 1 + std::rand() % 500;
        n[i] = 1 + std::rand() % 500;
        lda[i] = std::max(m[i], n[i]);
        incx[i] = -3 + std::rand() % 6;
        incx[i] = (incx[i] == 0) ? 3 : incx[i];
        incy[i] = -3 + std::rand() % 6;
        incy[i] = (incy[i] == 0) ? 3 : incy[i];
        alpha[i] = rand_scalar<fp>();
        beta[i] = rand_scalar<fp>();
        if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
            transa[i] = (oneapi::mkl::transpose)(std::rand() % 2);
        }
        else {
            tmp = std::rand() % 3;
            if (tmp == 2)
                transa[i] = oneapi::mkl::transpose::conjtrans;
            else
                transa[i] = (oneapi::mkl::transpose)tmp;
        }
        total_batch_count += group_size[i];
    }

    auto uafpp = usm_allocator<fp*, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp*, decltype(uafpp)> a_array(uafpp), x_array(uafpp), y_array(uafpp), y_ref_array(uafpp);
    a_array.resize(total_batch_count);
    x_array.resize(total_batch_count);
    y_array.resize(total_batch_count);
    y_ref_array.resize(total_batch_count);

    idx = 0;
    for (i = 0; i < group_count; i++) {
        size_a = (layout == oneapi::mkl::layout::col_major) ? lda[i] * n[i] : lda[i] * m[i];
        x_len = (transa[i] == oneapi::mkl::transpose::nontrans) ? n[i] : m[i];
        y_len = (transa[i] == oneapi::mkl::transpose::nontrans) ? m[i] : n[i];
        size_x = 1 + (x_len - 1) * std::abs(incx[i]);
        size_y = 1 + (y_len - 1) * std::abs(incy[i]);
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx] = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_a, *dev, cxt);
            x_array[idx] = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_x, *dev, cxt);
            y_array[idx] = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_y, *dev, cxt);
            y_ref_array[idx] = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * size_y, *dev, cxt);
            rand_matrix(a_array[idx], layout, oneapi::mkl::transpose::nontrans, m[i], n[i], lda[i]);
            rand_vector(x_array[idx], x_len, incx[i]);
            rand_vector(y_array[idx], y_len, incy[i]);
            copy_vector(y_array[idx], y_len, incy[i], y_ref_array[idx]);
            idx++;
        }
    }

    // Call reference GEMV_BATCH.
    using fp_ref = typename ref_type_info<fp>::type;
    int* m_ref = (int*)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int* n_ref = (int*)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int* lda_ref = (int*)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int* incx_ref = (int*)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int* incy_ref = (int*)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);
    int* group_size_ref = (int*)oneapi::mkl::aligned_alloc(64, sizeof(int) * group_count);

    CBLAS_TRANSPOSE* transa_ref =
        (CBLAS_TRANSPOSE*)oneapi::mkl::aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);

    if ((m_ref == NULL) || (n_ref == NULL) || (lda_ref == NULL) || (incx_ref == NULL) ||
        (incy_ref == NULL) || (transa_ref == NULL) || (group_size_ref == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        oneapi::mkl::aligned_free(m_ref);
        oneapi::mkl::aligned_free(n_ref);
        oneapi::mkl::aligned_free(lda_ref);
        oneapi::mkl::aligned_free(incx_ref);
        oneapi::mkl::aligned_free(incy_ref);
        oneapi::mkl::aligned_free(transa_ref);
        oneapi::mkl::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::mkl::free_shared(a_array[idx], cxt);
                oneapi::mkl::free_shared(x_array[idx], cxt);
                oneapi::mkl::free_shared(y_array[idx], cxt);
                oneapi::mkl::free_shared(y_ref_array[idx], cxt);
                idx++;
            }
        }
        return false;
    }
    idx = 0;
    for (i = 0; i < group_count; i++) {
        transa_ref[i] = convert_to_cblas_trans(transa[i]);
        m_ref[i] = (int)m[i];
        n_ref[i] = (int)n[i];
        lda_ref[i] = (int)lda[i];
        incx_ref[i] = (int)incx[i];
        incy_ref[i] = (int)incy[i];
        group_size_ref[i] = (int)group_size[i];
        for (j = 0; j < group_size_ref[i]; j++) {
            ::gemv(convert_to_cblas_layout(layout), transa_ref[i], (const int*)&m_ref[i],
                   (const int*)&n_ref[i], (const fp_ref*)&alpha[i], (const fp_ref*)a_array[idx],
                   (const int*)&lda_ref[i], (const fp_ref*)x_array[idx], (const int*)&incx_ref[i],
                   (const fp_ref*)&beta[i], (fp_ref*)y_ref_array[idx], (const int*)&incy_ref[i]);
            idx++;
        }
    }

    // Call DPC++ GEMV_BATCH.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                done = oneapi::mkl::blas::column_major::gemv_batch(
                    main_queue, &transa[0], &m[0], &n[0], &alpha[0], (const fp**)&a_array[0],
                    &lda[0], (const fp**)&x_array[0], &incx[0], &beta[0], &y_array[0], &incy[0],
                    group_count, &group_size[0], dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::gemv_batch(
                    main_queue, &transa[0], &m[0], &n[0], &alpha[0], (const fp**)&a_array[0],
                    &lda[0], (const fp**)&x_array[0], &incx[0], &beta[0], &y_array[0], &incy[0],
                    group_count, &group_size[0], dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::gemv_batch,
                                        &transa[0], &m[0], &n[0], &alpha[0],
                                        (const fp**)&a_array[0], &lda[0], (const fp**)&x_array[0],
                                        &incx[0], &beta[0], &y_array[0], &incy[0], group_count,
                                        &group_size[0], dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(
                    main_queue, oneapi::mkl::blas::row_major::gemv_batch, &transa[0], &m[0], &n[0],
                    &alpha[0], (const fp**)&a_array[0], &lda[0], (const fp**)&x_array[0], &incx[0],
                    &beta[0], &y_array[0], &incy[0], group_count, &group_size[0], dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMV_BATCH:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        oneapi::mkl::aligned_free(m_ref);
        oneapi::mkl::aligned_free(n_ref);
        oneapi::mkl::aligned_free(lda_ref);
        oneapi::mkl::aligned_free(incx_ref);
        oneapi::mkl::aligned_free(incy_ref);
        oneapi::mkl::aligned_free(transa_ref);
        oneapi::mkl::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::mkl::free_shared(a_array[idx], cxt);
                oneapi::mkl::free_shared(x_array[idx], cxt);
                oneapi::mkl::free_shared(y_array[idx], cxt);
                oneapi::mkl::free_shared(y_ref_array[idx], cxt);
                idx++;
            }
        }
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMV_BATCH:\n" << error.what() << std::endl;
    }

    bool good = true;
    // Compare the results of reference implementation and DPC++ implementation.
    idx = 0;
    for (i = 0; i < group_count; i++) {
        y_len = (transa[i] == oneapi::mkl::transpose::nontrans) ? m[i] : n[i];
        for (j = 0; j < group_size[i]; j++) {
            good = good && check_equal_vector(y_array[idx], y_ref_array[idx], y_len, incy[i],
                                              std::max<int>(m[i], n[i]), std::cout);
            idx++;
        }
    }

    oneapi::mkl::aligned_free(m_ref);
    oneapi::mkl::aligned_free(n_ref);
    oneapi::mkl::aligned_free(lda_ref);
    oneapi::mkl::aligned_free(incx_ref);
    oneapi::mkl::aligned_free(incy_ref);
    oneapi::mkl::aligned_free(transa_ref);
    oneapi::mkl::aligned_free(group_size_ref);
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            oneapi::mkl::free_shared(a_array[idx], cxt);
            oneapi::mkl::free_shared(x_array[idx], cxt);
            oneapi::mkl::free_shared(y_array[idx], cxt);
            oneapi::mkl::free_shared(y_ref_array[idx], cxt);
            idx++;
        }
    }

    return (int)good;
}

class GemvBatchUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(GemvBatchUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(GemvBatchUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(GemvBatchUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(GemvBatchUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(GemvBatchUsmTestSuite, GemvBatchUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
