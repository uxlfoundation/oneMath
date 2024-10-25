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
int test(device* dev, oneapi::mkl::layout layout, int64_t batch_size) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during SYRK_BATCH_STRIDE:\n"
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
    int64_t n, k;
    int64_t lda, ldc;
    oneapi::mkl::uplo upper_lower;
    oneapi::mkl::transpose trans;
    fp alpha, beta;

    int64_t i, tmp;

    batch_size = 1 + std::rand() % 20;
    n = 1 + std::rand() % 500;
    k = 1 + std::rand() % 500;
    lda = std::max(n, k);
    ldc = std::max(n, n);
    alpha = rand_scalar<fp>();
    beta = rand_scalar<fp>();
    upper_lower = (oneapi::mkl::uplo)(std::rand() % 2);
    if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
        trans = (std::rand() % 2) == 0   ? oneapi::mkl::transpose::nontrans
                : (std::rand() % 2) == 0 ? oneapi::mkl::transpose::trans
                                         : oneapi::mkl::transpose::conjtrans;
    }
    else {
        trans = (std::rand() % 2) == 0 ? oneapi::mkl::transpose::nontrans
                                       : oneapi::mkl::transpose::trans;
    }

    int64_t stride_a, stride_c;

    switch (layout) {
        case oneapi::mkl::layout::col_major:
            stride_a = (trans == oneapi::mkl::transpose::nontrans) ? lda * k : lda * n;
            stride_c = ldc * n;
            break;
        case oneapi::mkl::layout::row_major:
            stride_a = (trans == oneapi::mkl::transpose::nontrans) ? lda * n : lda * k;
            stride_c = ldc * n;
            break;
        default: break;
    }

    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> A(ua), C(ua), C_ref(ua);

    A.resize(stride_a * batch_size);
    C.resize(stride_c * batch_size);
    C_ref.resize(stride_c * batch_size);

    fp** a_array = (fp**)oneapi::mkl::malloc_shared(64, sizeof(fp*) * batch_size, *dev, cxt);
    fp** c_array = (fp**)oneapi::mkl::malloc_shared(64, sizeof(fp*) * batch_size, *dev, cxt);
    fp** c_ref_array = (fp**)oneapi::mkl::malloc_shared(64, sizeof(fp*) * batch_size, *dev, cxt);

    if ((a_array == NULL) || (c_array == NULL) || (c_ref_array == NULL)) {
        std::cout << "Error cannot allocate arrays of pointers\n";
        oneapi::mkl::free_shared(a_array, cxt);
        oneapi::mkl::free_shared(c_array, cxt);
        oneapi::mkl::free_shared(c_ref_array, cxt);
        return false;
    }

    for (i = 0; i < batch_size; i++) {
        a_array[i] = &A[i * stride_a];
        c_array[i] = &C[i * stride_c];
        c_ref_array[i] = &C_ref[i * stride_c];
    }

    rand_matrix(A, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_a * batch_size, 1, stride_a * batch_size);
    rand_matrix(C, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_c * batch_size, 1, stride_c * batch_size);
    copy_matrix(C, oneapi::mkl::layout::col_major, oneapi::mkl::transpose::nontrans,
                stride_c * batch_size, 1, stride_c * batch_size, C_ref);

    // Call reference SYRK_BATCH_STRIDE.
    using fp_ref = typename ref_type_info<fp>::type;
    int n_ref = (int)n;
    int k_ref = (int)k;
    int lda_ref = (int)lda;
    int ldc_ref = (int)ldc;
    int batch_size_ref = (int)batch_size;
    for (i = 0; i < batch_size_ref; i++) {
        ::syrk(convert_to_cblas_layout(layout), convert_to_cblas_uplo(upper_lower),
               convert_to_cblas_trans(trans), (const int*)&n_ref, (const int*)&k_ref,
               (const fp_ref*)&alpha, (const fp_ref*)(A.data() + stride_a * i),
               (const int*)&lda_ref, (const fp_ref*)&beta, (fp_ref*)(C_ref.data() + stride_c * i),
               (const int*)&ldc_ref);
    }

    // Call DPC++ SYRK_BATCH_STRIDE.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                done = oneapi::mkl::blas::column_major::syrk_batch(
                    main_queue, upper_lower, trans, n, k, alpha, &A[0], lda, stride_a, beta, &C[0],
                    ldc, stride_c, batch_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::syrk_batch(
                    main_queue, upper_lower, trans, n, k, alpha, &A[0], lda, stride_a, beta, &C[0],
                    ldc, stride_c, batch_size, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::syrk_batch,
                                        upper_lower, trans, n, k, alpha, &A[0], lda, stride_a, beta,
                                        &C[0], ldc, stride_c, batch_size, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::syrk_batch,
                                        upper_lower, trans, n, k, alpha, &A[0], lda, stride_a, beta,
                                        &C[0], ldc, stride_c, batch_size, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during SYRK_BATCH_STRIDE:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        oneapi::mkl::free_shared(a_array, cxt);
        oneapi::mkl::free_shared(c_array, cxt);
        oneapi::mkl::free_shared(c_ref_array, cxt);
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of SYRK_BATCH_STRIDE:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = check_equal_matrix(C, C_ref, oneapi::mkl::layout::col_major, stride_c * batch_size,
                                   1, stride_c * batch_size, 10 * k, std::cout);

    oneapi::mkl::free_shared(a_array, cxt);
    oneapi::mkl::free_shared(c_array, cxt);
    oneapi::mkl::free_shared(c_ref_array, cxt);

    return (int)good;
}

class SyrkBatchStrideUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(SyrkBatchStrideUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchStrideUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchStrideUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchStrideUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(SyrkBatchStrideUsmTestSuite, SyrkBatchStrideUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
