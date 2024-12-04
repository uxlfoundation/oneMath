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
#include <type_traits>

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
int test(device* dev, oneapi::math::layout layout, int64_t group_count) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during OMATCOPY_BATCH:\n"
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
    vector<int64_t, decltype(uaint)> m(uaint), n(uaint), lda(uaint), ldb(uaint), group_size(uaint);

    auto uatranspose = usm_allocator<oneapi::math::transpose, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::math::transpose, decltype(uatranspose)> trans(uatranspose);

    auto uafp = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(uafp)> alpha(uafp);

    m.resize(group_count);
    n.resize(group_count);
    lda.resize(group_count);
    ldb.resize(group_count);
    group_size.resize(group_count);
    trans.resize(group_count);
    alpha.resize(group_count);

    int64_t i, tmp;
    int64_t j, idx = 0;
    int64_t total_batch_count = 0;
    int64_t size_a = 0, size_b = 0;

    for (i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 20;
        m[i] = 1 + std::rand() % 50;
        n[i] = 1 + std::rand() % 50;
        lda[i] = std::max(m[i], n[i]);
        ldb[i] = std::max(m[i], n[i]);
        alpha[i] = rand_scalar<fp>();
        trans[i] = rand_trans<fp>();
        total_batch_count += group_size[i];
    }

    auto uafpp = usm_allocator<fp*, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp*, decltype(uafpp)> a_array(uafpp), b_array(uafpp), b_ref_array(uafpp);

    a_array.resize(total_batch_count);
    b_array.resize(total_batch_count);
    b_ref_array.resize(total_batch_count);

    idx = 0;
    for (i = 0; i < group_count; i++) {
        switch (layout) {
            case oneapi::math::layout::col_major:
                size_a = lda[i] * n[i];
                size_b =
                    (trans[i] == oneapi::math::transpose::nontrans) ? ldb[i] * n[i] : ldb[i] * m[i];
                break;
            case oneapi::math::layout::row_major:
                size_a = lda[i] * m[i];
                size_b =
                    (trans[i] == oneapi::math::transpose::nontrans) ? ldb[i] * m[i] : ldb[i] * n[i];
                break;
            default: break;
        }
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx] = (fp*)oneapi::math::malloc_shared(64, sizeof(fp) * size_a, *dev, cxt);
            b_array[idx] = (fp*)oneapi::math::malloc_shared(64, sizeof(fp) * size_b, *dev, cxt);
            b_ref_array[idx] = (fp*)oneapi::math::malloc_shared(64, sizeof(fp) * size_b, *dev, cxt);
            rand_matrix(a_array[idx], oneapi::math::layout::col_major,
                        oneapi::math::transpose::nontrans, size_a, 1, size_a);
            rand_matrix(b_array[idx], oneapi::math::layout::col_major,
                        oneapi::math::transpose::nontrans, size_b, 1, size_b);
            copy_matrix(b_array[idx], oneapi::math::layout::col_major,
                        oneapi::math::transpose::nontrans, size_b, 1, size_b, b_ref_array[idx]);
            idx++;
        }
    }

    // Call reference OMATCOPY
    idx = 0;
    for (i = 0; i < group_count; i++) {
        int m_ref = (int)m[i];
        int n_ref = (int)n[i];
        int lda_ref = (int)lda[i];
        int ldb_ref = (int)ldb[i];
        int group_size_ref = (int)group_size[i];
        for (j = 0; j < group_size_ref; j++) {
            omatcopy_ref(layout, trans[i], m_ref, n_ref, alpha[i], a_array[idx], lda_ref,
                         b_ref_array[idx], ldb_ref);
            idx++;
        }
    }

    // Call DPC++ OMATCOPY_BATCH
    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                done = oneapi::math::blas::column_major::omatcopy_batch(
                    main_queue, trans.data(), m.data(), n.data(), alpha.data(),
                    (const fp**)a_array.data(), lda.data(), b_array.data(), ldb.data(), group_count,
                    group_size.data(), dependencies);
                break;
            case oneapi::math::layout::row_major:
                done = oneapi::math::blas::row_major::omatcopy_batch(
                    main_queue, trans.data(), m.data(), n.data(), alpha.data(),
                    (const fp**)a_array.data(), lda.data(), b_array.data(), ldb.data(), group_count,
                    group_size.data(), dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(
                    main_queue, oneapi::math::blas::column_major::omatcopy_batch, trans.data(),
                    m.data(), n.data(), alpha.data(), (const fp**)a_array.data(), lda.data(),
                    b_array.data(), ldb.data(), group_count, group_size.data(), dependencies);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::omatcopy_batch,
                                        trans.data(), m.data(), n.data(), alpha.data(),
                                        (const fp**)a_array.data(), lda.data(), b_array.data(),
                                        ldb.data(), group_count, group_size.data(), dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during OMATCOPY_BATCH:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::math::free_shared(a_array[idx], cxt);
                oneapi::math::free_shared(b_array[idx], cxt);
                oneapi::math::free_shared(b_ref_array[idx], cxt);
                idx++;
            }
        }
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of OMATCOPY_BATCH:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = true;
    idx = 0;
    for (i = 0; i < group_count; i++) {
        switch (layout) {
            case oneapi::math::layout::col_major:
                size_a = lda[i] * n[i];
                size_b =
                    (trans[i] == oneapi::math::transpose::nontrans) ? ldb[i] * n[i] : ldb[i] * m[i];
                break;
            case oneapi::math::layout::row_major:
                size_a = lda[i] * m[i];
                size_b =
                    (trans[i] == oneapi::math::transpose::nontrans) ? ldb[i] * m[i] : ldb[i] * n[i];
                break;
            default: break;
        }
        for (j = 0; j < group_size[i]; j++) {
            good = good && check_equal_matrix(b_array[idx], b_ref_array[idx],
                                              oneapi::math::layout::col_major, size_b, 1, size_b,
                                              10, std::cout);
            idx++;
        }
    }

    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            oneapi::math::free_shared(a_array[idx], cxt);
            oneapi::math::free_shared(b_array[idx], cxt);
            oneapi::math::free_shared(b_ref_array[idx], cxt);
            idx++;
        }
    }

    return (int)good;
}

class OmatcopyBatchUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {};

TEST_P(OmatcopyBatchUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmatcopyBatchUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmatcopyBatchUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(OmatcopyBatchUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(OmatcopyBatchUsmTestSuite, OmatcopyBatchUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
