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
#include "oneapi/math.hpp"
#include "oneapi/math/detail/config.hpp"
#include "allocator_helper.hpp"
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
int test(device* dev, oneapi::math::layout layout, int64_t group_count) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during SYRK_BATCH:\n"
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
    vector<int64_t, decltype(uaint)> n(uaint), k(uaint), lda(uaint), ldc(uaint), group_size(uaint);

    auto uauplo = usm_allocator<oneapi::math::uplo, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::math::uplo, decltype(uauplo)> upper_lower(uauplo);

    auto uatranspose = usm_allocator<oneapi::math::transpose, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::math::transpose, decltype(uatranspose)> trans(uatranspose);

    auto uafp = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(uafp)> alpha(uafp), beta(uafp);

    n.resize(group_count);
    k.resize(group_count);
    lda.resize(group_count);
    ldc.resize(group_count);
    group_size.resize(group_count);
    trans.resize(group_count);
    upper_lower.resize(group_count);
    alpha.resize(group_count);
    beta.resize(group_count);

    int64_t i, tmp;
    int64_t j, idx = 0;
    int64_t total_batch_count = 0;
    int64_t size_a = 0, size_c = 0;

    for (i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 20;
        n[i] = 1 + std::rand() % 500;
        k[i] = 1 + std::rand() % 500;
        lda[i] = std::max(n[i], k[i]);
        ldc[i] = std::max(n[i], n[i]);
        alpha[i] = rand_scalar<fp>();
        beta[i] = rand_scalar<fp>();
        upper_lower[i] = (oneapi::math::uplo)(std::rand() % 2);
        if ((std::is_same<fp, float>::value) || (std::is_same<fp, double>::value)) {
            trans[i] = (std::rand() % 2) == 0   ? oneapi::math::transpose::nontrans
                       : (std::rand() % 2) == 0 ? oneapi::math::transpose::trans
                                                : oneapi::math::transpose::conjtrans;
        }
        else {
            trans[i] = (std::rand() % 2) == 0 ? oneapi::math::transpose::nontrans
                                              : oneapi::math::transpose::trans;
        }
        total_batch_count += group_size[i];
    }

    auto uafpp = usm_allocator<fp*, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp*, decltype(uafpp)> a_array(uafpp), c_array(uafpp), c_ref_array(uafpp);
    a_array.resize(total_batch_count);
    c_array.resize(total_batch_count);
    c_ref_array.resize(total_batch_count);

    idx = 0;
    for (i = 0; i < group_count; i++) {
        switch (layout) {
            case oneapi::math::layout::col_major:
                size_a = lda[i] * ((trans[i] == oneapi::math::transpose::nontrans) ? k[i] : n[i]);
                size_c = ldc[i] * n[i];
                break;
            case oneapi::math::layout::row_major:
                size_a = lda[i] * ((trans[i] == oneapi::math::transpose::nontrans) ? n[i] : k[i]);
                size_c = ldc[i] * n[i];
                break;
            default: break;
        }
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx] = (fp*)oneapi::math::malloc_shared(64, sizeof(fp) * size_a, *dev, cxt);
            c_array[idx] = (fp*)oneapi::math::malloc_shared(64, sizeof(fp) * size_c, *dev, cxt);
            c_ref_array[idx] = (fp*)oneapi::math::malloc_shared(64, sizeof(fp) * size_c, *dev, cxt);
            rand_matrix(a_array[idx], layout, trans[i], n[i], k[i], lda[i]);
            rand_matrix(c_array[idx], layout, oneapi::math::transpose::nontrans, n[i], n[i],
                        ldc[i]);
            copy_matrix(c_array[idx], layout, oneapi::math::transpose::nontrans, n[i], n[i], ldc[i],
                        c_ref_array[idx]);
            idx++;
        }
    }

    // Call reference SYRK_BATCH.
    using fp_ref = typename ref_type_info<fp>::type;
    int* n_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* k_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* lda_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* ldc_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* group_size_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);

    CBLAS_UPLO* upper_lower_ref =
        (CBLAS_UPLO*)oneapi::math::aligned_alloc(64, sizeof(CBLAS_UPLO) * group_count);
    CBLAS_TRANSPOSE* trans_ref =
        (CBLAS_TRANSPOSE*)oneapi::math::aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);

    if ((n_ref == NULL) || (k_ref == NULL) || (lda_ref == NULL) || (ldc_ref == NULL) ||
        (trans_ref == NULL) || (upper_lower_ref == NULL) || (group_size_ref == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        oneapi::math::aligned_free(n_ref);
        oneapi::math::aligned_free(k_ref);
        oneapi::math::aligned_free(lda_ref);
        oneapi::math::aligned_free(ldc_ref);
        oneapi::math::aligned_free(trans_ref);
        oneapi::math::aligned_free(upper_lower_ref);
        oneapi::math::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::math::free_shared(a_array[idx], cxt);
                oneapi::math::free_shared(c_array[idx], cxt);
                oneapi::math::free_shared(c_ref_array[idx], cxt);
                idx++;
            }
        }
        return false;
    }
    idx = 0;
    for (i = 0; i < group_count; i++) {
        trans_ref[i] = convert_to_cblas_trans(trans[i]);
        upper_lower_ref[i] = convert_to_cblas_uplo(upper_lower[i]);
        n_ref[i] = (int)n[i];
        k_ref[i] = (int)k[i];
        lda_ref[i] = (int)lda[i];
        ldc_ref[i] = (int)ldc[i];
        group_size_ref[i] = (int)group_size[i];
        for (j = 0; j < group_size_ref[i]; j++) {
            ::syrk(convert_to_cblas_layout(layout), upper_lower_ref[i], trans_ref[i],
                   (const int*)&n_ref[i], (const int*)&k_ref[i], (const fp_ref*)&alpha[i],
                   (const fp_ref*)a_array[idx], (const int*)&lda_ref[i], (const fp_ref*)&beta[i],
                   (fp_ref*)c_ref_array[idx], (const int*)&ldc_ref[i]);
            idx++;
        }
    }

    // Call DPC++ SYRK_BATCH.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::syrk_batch(
                    main_queue, &upper_lower[0], &trans[0], &n[0], &k[0], &alpha[0],
                    (const fp**)&a_array[0], &lda[0], &beta[0], &c_array[0], &ldc[0], group_count,
                    &group_size[0], dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::syrk_batch(
                    main_queue, &upper_lower[0], &trans[0], &n[0], &k[0], &alpha[0],
                    (const fp**)&a_array[0], &lda[0], &beta[0], &c_array[0], &ldc[0], group_count,
                    &group_size[0], dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::syrk_batch,
                                        &upper_lower[0], &trans[0], &n[0], &k[0], &alpha[0],
                                        (const fp**)&a_array[0], &lda[0], &beta[0], &c_array[0],
                                        &ldc[0], group_count, &group_size[0], dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::syrk_batch,
                                        &upper_lower[0], &trans[0], &n[0], &k[0], &alpha[0],
                                        (const fp**)&a_array[0], &lda[0], &beta[0], &c_array[0],
                                        &ldc[0], group_count, &group_size[0], dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during SYRK_BATCH:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        oneapi::math::aligned_free(n_ref);
        oneapi::math::aligned_free(k_ref);
        oneapi::math::aligned_free(lda_ref);
        oneapi::math::aligned_free(ldc_ref);
        oneapi::math::aligned_free(upper_lower_ref);
        oneapi::math::aligned_free(trans_ref);
        oneapi::math::aligned_free(group_size_ref);
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::math::free_shared(a_array[idx], cxt);
                oneapi::math::free_shared(c_array[idx], cxt);
                oneapi::math::free_shared(c_ref_array[idx], cxt);
                idx++;
            }
        }
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of SYRK_BATCH:\n" << error.what() << std::endl;
    }

    bool good = true;
    // Compare the results of reference implementation and DPC++ implementation.
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            good = good && check_equal_matrix(c_array[idx], c_ref_array[idx], layout, n[i], n[i],
                                              ldc[i], 10 * k[i], std::cout);
            idx++;
        }
    }
    oneapi::math::aligned_free(n_ref);
    oneapi::math::aligned_free(k_ref);
    oneapi::math::aligned_free(lda_ref);
    oneapi::math::aligned_free(ldc_ref);
    oneapi::math::aligned_free(upper_lower_ref);
    oneapi::math::aligned_free(trans_ref);
    oneapi::math::aligned_free(group_size_ref);
    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            oneapi::math::free_shared(a_array[idx], cxt);
            oneapi::math::free_shared(c_array[idx], cxt);
            oneapi::math::free_shared(c_ref_array[idx], cxt);
            idx++;
        }
    }

    return (int)good;
}

class SyrkBatchUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(SyrkBatchUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(SyrkBatchUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(SyrkBatchUsmTestSuite, SyrkBatchUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
