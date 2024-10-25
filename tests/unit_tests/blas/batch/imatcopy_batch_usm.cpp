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
                std::cout << "Caught asynchronous SYCL exception during IMATCOPY_BATCH:\n"
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

    auto uatranspose = usm_allocator<oneapi::mkl::transpose, usm::alloc::shared, 64>(cxt, *dev);
    vector<oneapi::mkl::transpose, decltype(uatranspose)> trans(uatranspose);

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
    int64_t size_a = 0, size_b = 0, size = 0;

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
    vector<fp*, decltype(uafpp)> ab_array(uafpp), ab_ref_array(uafpp);

    ab_array.resize(total_batch_count);
    ab_ref_array.resize(total_batch_count);

    idx = 0;
    for (i = 0; i < group_count; i++) {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                size_a = lda[i] * n[i];
                size_b =
                    (trans[i] == oneapi::mkl::transpose::nontrans) ? ldb[i] * n[i] : ldb[i] * m[i];
                break;
            case oneapi::mkl::layout::row_major:
                size_a = lda[i] * m[i];
                size_b =
                    (trans[i] == oneapi::mkl::transpose::nontrans) ? ldb[i] * m[i] : ldb[i] * n[i];
                break;
            default: break;
        }
        size = std::max(size_a, size_b);
        for (j = 0; j < group_size[i]; j++) {
            ab_array[idx] = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * size, *dev, cxt);
            ab_ref_array[idx] = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp) * size, *dev, cxt);
            rand_matrix(ab_array[idx], oneapi::mkl::layout::col_major,
                        oneapi::mkl::transpose::nontrans, size, 1, size);
            copy_matrix(ab_array[idx], oneapi::mkl::layout::col_major,
                        oneapi::mkl::transpose::nontrans, size, 1, size, ab_ref_array[idx]);
            idx++;
        }
    }

    // Call reference IMATCOPY
    idx = 0;
    for (i = 0; i < group_count; i++) {
        int m_ref = (int)m[i];
        int n_ref = (int)n[i];
        int lda_ref = (int)lda[i];
        int ldb_ref = (int)ldb[i];
        int group_size_ref = (int)group_size[i];
        for (j = 0; j < group_size_ref; j++) {
            imatcopy_ref(layout, trans[i], m_ref, n_ref, alpha[i], ab_ref_array[idx], lda_ref,
                         ldb_ref);
            idx++;
        }
    }

    // Call DPC++ IMATCOPY_BATCH
    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                done = oneapi::mkl::blas::column_major::imatcopy_batch(
                    main_queue, trans.data(), m.data(), n.data(), alpha.data(), ab_array.data(),
                    lda.data(), ldb.data(), group_count, group_size.data(), dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::imatcopy_batch(
                    main_queue, trans.data(), m.data(), n.data(), alpha.data(), ab_array.data(),
                    lda.data(), ldb.data(), group_count, group_size.data(), dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::imatcopy_batch,
                                        trans.data(), m.data(), n.data(), alpha.data(),
                                        ab_array.data(), lda.data(), ldb.data(), group_count,
                                        group_size.data(), dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::imatcopy_batch,
                                        trans.data(), m.data(), n.data(), alpha.data(),
                                        ab_array.data(), lda.data(), ldb.data(), group_count,
                                        group_size.data(), dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during IMATCOPY_BATCH:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        idx = 0;
        for (i = 0; i < group_count; i++) {
            for (j = 0; j < group_size[i]; j++) {
                oneapi::mkl::free_shared(ab_array[idx], cxt);
                oneapi::mkl::free_shared(ab_ref_array[idx], cxt);
                idx++;
            }
        }
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of IMATCOPY_BATCH:\n"
                  << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    bool good = true;
    idx = 0;
    for (i = 0; i < group_count; i++) {
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                size_a = lda[i] * n[i];
                size_b =
                    (trans[i] == oneapi::mkl::transpose::nontrans) ? ldb[i] * n[i] : ldb[i] * m[i];
                break;
            case oneapi::mkl::layout::row_major:
                size_a = lda[i] * m[i];
                size_b =
                    (trans[i] == oneapi::mkl::transpose::nontrans) ? ldb[i] * m[i] : ldb[i] * n[i];
                break;
            default: break;
        }
        size = std::max(size_a, size_b);
        for (j = 0; j < group_size[i]; j++) {
            good = good &&
                   check_equal_matrix(ab_array[idx], ab_ref_array[idx],
                                      oneapi::mkl::layout::col_major, size, 1, size, 10, std::cout);
            idx++;
        }
    }

    idx = 0;
    for (i = 0; i < group_count; i++) {
        for (j = 0; j < group_size[i]; j++) {
            oneapi::mkl::free_shared(ab_array[idx], cxt);
            oneapi::mkl::free_shared(ab_ref_array[idx], cxt);
            idx++;
        }
    }

    return (int)good;
}

class ImatcopyBatchUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(ImatcopyBatchUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        test<std::complex<float>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

TEST_P(ImatcopyBatchUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        test<std::complex<double>>(std::get<0>(GetParam()), std::get<1>(GetParam()), 5));
}

INSTANTIATE_TEST_SUITE_P(ImatcopyBatchUsmTestSuite, ImatcopyBatchUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
