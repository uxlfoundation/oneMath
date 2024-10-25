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
int test(device* dev, oneapi::mkl::layout layout, int64_t group_count) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during AXPY_BATCH:\n"
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
    int64_t* n = (int64_t*)oneapi::mkl::malloc_shared(64, sizeof(int64_t) * group_count, *dev, cxt);
    int64_t* incx =
        (int64_t*)oneapi::mkl::malloc_shared(64, sizeof(int64_t) * group_count, *dev, cxt);
    int64_t* incy =
        (int64_t*)oneapi::mkl::malloc_shared(64, sizeof(int64_t) * group_count, *dev, cxt);
    fp* alpha = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * group_count, *dev, cxt);
    int64_t* group_size =
        (int64_t*)oneapi::mkl::malloc_shared(64, sizeof(int64_t) * group_count, *dev, cxt);

    if ((n == NULL) || (incx == NULL) || (incy == NULL) || (alpha == NULL) ||
        (group_size == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        oneapi::mkl::free_shared(n, cxt);
        oneapi::mkl::free_shared(incx, cxt);
        oneapi::mkl::free_shared(incy, cxt);
        oneapi::mkl::free_shared(alpha, cxt);
        oneapi::mkl::free_shared(group_size, cxt);
        return false;
    }

    int64_t i;
    int64_t j, idx = 0;
    int64_t total_size_x, total_size_y;
    int64_t total_batch_count = 0;

    for (i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 100;
        n[i] = 1 + std::rand() % 500;
        incx[i] = ((std::rand() % 2) == 0) ? 1 + std::rand() % 2 : -1 - std::rand() % 2;
        incy[i] = ((std::rand() % 2) == 0) ? 1 + std::rand() % 2 : -1 - std::rand() % 2;
        alpha[i] = rand_scalar<fp>();
        total_batch_count += group_size[i];
    }

    fp** x_array = (fp**)oneapi::mkl::malloc_shared(64, sizeof(fp*) * total_batch_count, *dev, cxt);
    fp** y_array = (fp**)oneapi::mkl::malloc_shared(64, sizeof(fp*) * total_batch_count, *dev, cxt);
    fp** y_ref_array =
        (fp**)oneapi::mkl::malloc_shared(64, sizeof(fp*) * total_batch_count, *dev, cxt);

    if ((x_array == NULL) || (y_array == NULL) || (y_ref_array == NULL)) {
        std::cout << "Error cannot allocate arrays of pointers\n";
        oneapi::mkl::free_shared(x_array, cxt);
        oneapi::mkl::free_shared(y_array, cxt);
        oneapi::mkl::free_shared(y_ref_array, cxt);
        return false;
    }
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            total_size_x = (1 + (n[i] - 1) * std::abs(incx[i]));
            total_size_y = (1 + (n[i] - 1) * std::abs(incy[i]));
            x_array[idx] =
                (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * total_size_x, *dev, cxt);
            y_array[idx] =
                (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * total_size_y, *dev, cxt);
            y_ref_array[idx] =
                (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * total_size_y, *dev, cxt);
            rand_vector(x_array[idx], n[i], incx[i]);
            rand_vector(y_array[idx], n[i], incy[i]);
            copy_vector(y_array[idx], n[i], incy[i], y_ref_array[idx]);
            idx++;
        }
    }

    // Call reference AXPY_BATCH.
    using fp_ref = typename ref_type_info<fp>::type;
    int n_ref, incx_ref, incy_ref;

    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            n_ref = (int)n[i];
            incx_ref = (int)incx[i];
            incy_ref = (int)incy[i];
            ::axpy((const int*)&n_ref, (const fp_ref*)&alpha[i], (const fp_ref*)x_array[idx],
                   (const int*)&incx_ref, (fp_ref*)y_ref_array[idx], (const int*)&incy_ref);
            idx++;
        }
    }

    // Call DPC++ AXPY_BATCH.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                done = oneapi::mkl::blas::column_major::axpy_batch(
                    main_queue, n, alpha, (const fp**)x_array, incx, y_array, incy, group_count,
                    group_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::axpy_batch(
                    main_queue, n, alpha, (const fp**)x_array, incx, y_array, incy, group_count,
                    group_size, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::axpy_batch, n,
                                        alpha, (const fp**)x_array, incx, y_array, incy,
                                        group_count, group_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::axpy_batch, n,
                                        alpha, (const fp**)x_array, incx, y_array, incy,
                                        group_count, group_size, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during AXPY_BATCH:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::mkl::free_shared(x_array[idx], cxt);
                oneapi::mkl::free_shared(y_array[idx], cxt);
                oneapi::mkl::free_shared(y_ref_array[idx], cxt);
                idx++;
            }
        }
        oneapi::mkl::free_shared(n, cxt);
        oneapi::mkl::free_shared(incx, cxt);
        oneapi::mkl::free_shared(incy, cxt);
        oneapi::mkl::free_shared(alpha, cxt);
        oneapi::mkl::free_shared(group_size, cxt);
        oneapi::mkl::free_shared(x_array, cxt);
        oneapi::mkl::free_shared(y_array, cxt);
        oneapi::mkl::free_shared(y_ref_array, cxt);
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of AXPY_BATCH:\n" << error.what() << std::endl;
    }

    bool good = true;

    // Compare the results of reference implementation and DPC++ implementation.
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            good = good && check_equal_vector(y_array[idx], y_ref_array[idx], n[i], incy[i], n[i],
                                              std::cout);
            idx++;
        }
    }

    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            oneapi::mkl::free_shared(x_array[idx], cxt);
            oneapi::mkl::free_shared(y_array[idx], cxt);
            oneapi::mkl::free_shared(y_ref_array[idx], cxt);
            idx++;
        }
    }
    oneapi::mkl::free_shared(n, cxt);
    oneapi::mkl::free_shared(incx, cxt);
    oneapi::mkl::free_shared(incy, cxt);
    oneapi::mkl::free_shared(alpha, cxt);
    oneapi::mkl::free_shared(group_size, cxt);
    oneapi::mkl::free_shared(x_array, cxt);
    oneapi::mkl::free_shared(y_array, cxt);
    oneapi::mkl::free_shared(y_ref_array, cxt);

    return (int)good;
}

class AxpyBatchUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(AxpyBatchUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(AxpyBatchUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(AxpyBatchUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(AxpyBatchUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(AxpyBatchUsmTestSuite, AxpyBatchUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
