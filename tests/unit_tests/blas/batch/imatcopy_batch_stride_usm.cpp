/*******************************************************************************
* Copyright 2022 Intel Corporation
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

template <typename fp>
int test(device* dev, oneapi::math::layout layout, int64_t batch_size) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during OMATCOPY_BATCH_STRIDE:\n"
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
    int64_t m, n;
    int64_t lda, ldb;
    oneapi::math::transpose trans;
    fp alpha;
    int64_t i;

    batch_size = 1 + std::rand() % 20;
    m = 1 + std::rand() % 50;
    n = 1 + std::rand() % 50;
    lda = std::max(m, n);
    ldb = std::max(m, n);
    alpha = rand_scalar<fp>();
    trans = rand_trans<fp>();

    int64_t stride_a, stride_b, stride;
    switch (layout) {
        case oneapi::math::layout::col_major:
            stride_a = lda * n;
            stride_b = (trans == oneapi::math::transpose::nontrans) ? ldb * n : ldb * m;
            stride = std::max(stride_a, stride_b);
            break;
        case oneapi::math::layout::row_major:
            stride_a = lda * m;
            stride_b = (trans == oneapi::math::transpose::nontrans) ? ldb * m : ldb * n;
            stride = std::max(stride_a, stride_b);
            break;
        default: break;
    }

    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> AB(ua), AB_ref(ua);

    AB.resize(stride * batch_size);
    AB_ref.resize(stride * batch_size);
    fp** ab_array = (fp**)oneapi::math::malloc_shared(64, sizeof(fp*) * batch_size, *dev, cxt);
    fp** ab_ref_array = (fp**)oneapi::math::malloc_shared(64, sizeof(fp*) * batch_size, *dev, cxt);
    if ((ab_array == NULL) || (ab_ref_array == NULL)) {
        std::cout << "Error cannot allocate arrays of pointers\n";
        oneapi::math::free_shared(ab_array, cxt);
        oneapi::math::free_shared(ab_ref_array, cxt);
        return false;
    }

    for (i = 0; i < batch_size; i++) {
        ab_array[i] = &AB[i * stride];
        ab_ref_array[i] = &AB_ref[i * stride];
    }

    rand_matrix(AB, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride * batch_size, 1, stride * batch_size);
    copy_matrix(AB, oneapi::math::layout::col_major, oneapi::math::transpose::nontrans,
                stride * batch_size, 1, stride * batch_size, AB_ref);

    // Call reference IMATCOPY_BATCH_STRIDE.
    int m_ref = (int)m;
    int n_ref = (int)n;
    int lda_ref = (int)lda;
    int ldb_ref = (int)ldb;
    int batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        imatcopy_ref(layout, trans, m_ref, n_ref, alpha, ab_ref_array[i], lda_ref, ldb_ref);
    }

    // Call DPC++ IMATCOPY_BATCH_STRIDE
    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::imatcopy_batch(
                    main_queue, trans, m, n, alpha, &AB[0], lda, ldb, stride, batch_size,
                    dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::imatcopy_batch(main_queue, trans, m, n, alpha,
                                                                     &AB[0], lda, ldb, stride,
                                                                     batch_size, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(
                    main_queue, oneapi::math::blas::column_major::imatcopy_batch, trans, m, n,
                    alpha, &AB[0], lda, ldb, stride, batch_size, dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::imatcopy_batch,
                                        trans, m, n, alpha, &AB[0], lda, ldb, stride, batch_size,
                                        dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during IMATCOPY_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        oneapi::math::free_shared(ab_array, cxt);
        oneapi::math::free_shared(ab_ref_array, cxt);
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of IMATCOPY_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_matrix(AB, AB_ref, oneapi::math::layout::col_major, stride * batch_size,
                                   1, stride * batch_size, 10, std::cout);

    oneapi::math::free_shared(ab_array, cxt);
    oneapi::math::free_shared(ab_ref_array, cxt);

    return (int)good;
}

class ImatcopyBatchStrideUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(ImatcopyBatchStrideUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchStrideUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchStrideUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchStrideUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(ImatcopyBatchStrideUsmTestSuite, ImatcopyBatchStrideUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
