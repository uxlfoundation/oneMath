/*******************************************************************************
* Copyright 2025 Intel Corporation
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

extern std::vector<sycl::device*> devices;

namespace {

#ifdef SYCL_EXT_ONEAPI_GRAPH
template <typename Ta, typename Tb, typename Tc, typename Ts>
int test(device* dev, oneapi::math::layout layout, int64_t group_count, size_t graph_nodes) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM_BATCH:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler, property::queue::in_order{});
    context cxt = main_queue.get_context();

    // Prepare data.
    auto uaint = usm_allocator<int64_t, usm::alloc::shared, 64>(cxt, *dev);
    std::vector<int64_t, decltype(uaint)> m(uaint), n(uaint), k(uaint), lda(uaint), ldb(uaint),
        ldc(uaint), group_size(uaint);

    auto uatranspose = usm_allocator<oneapi::math::transpose, usm::alloc::shared, 64>(cxt, *dev);
    std::vector<oneapi::math::transpose, decltype(uatranspose)> transa(uatranspose),
        transb(uatranspose);

    auto uaTs = usm_allocator<Ts, usm::alloc::shared, 64>(cxt, *dev);
    std::vector<Ts, decltype(uaTs)> alpha(uaTs), beta(uaTs);

    m.resize(group_count);
    n.resize(group_count);
    k.resize(group_count);
    lda.resize(group_count);
    ldb.resize(group_count);
    ldc.resize(group_count);
    group_size.resize(group_count);
    transa.resize(group_count);
    transb.resize(group_count);
    alpha.resize(group_count);
    beta.resize(group_count);

    int64_t total_batch_count = 0;
    for (int64_t i = 0; i < group_count; i++) {
        group_size[i] = 1 + std::rand() % 20;
        m[i] = 1 + std::rand() % 500;
        n[i] = 1 + std::rand() % 500;
        k[i] = 1 + std::rand() % 500;
        lda[i] = std::max(m[i], k[i]);
        ldb[i] = std::max(n[i], k[i]);
        ldc[i] = std::max(m[i], n[i]);
        alpha[i] = rand_scalar<Ts>();
        beta[i] = rand_scalar<Ts>();
        transa[i] = (oneapi::math::transpose)(std::rand() % 2);
        transb[i] = (oneapi::math::transpose)(std::rand() % 2);
        total_batch_count += group_size[i];
    }

    auto uaTap = usm_allocator<Ta*, usm::alloc::shared, 64>(cxt, *dev);
    auto uaTbp = usm_allocator<Tb*, usm::alloc::shared, 64>(cxt, *dev);
    auto uaTcp = usm_allocator<Tc*, usm::alloc::shared, 64>(cxt, *dev);
    auto uaTsp = usm_allocator<Ts*, usm::alloc::shared, 64>(cxt, *dev);
    std::vector<Ta*, decltype(uaTap)> a_array(uaTap);
    std::vector<Tb*, decltype(uaTbp)> b_array(uaTbp);
    std::vector<Tc*, decltype(uaTcp)> c_array(uaTcp), c_cast_ref_array(uaTcp);
    std::vector<Ts*, decltype(uaTsp)> a_ref_array(uaTsp), b_ref_array(uaTsp), c_ref_array(uaTsp);
    a_array.resize(total_batch_count);
    b_array.resize(total_batch_count);
    c_array.resize(total_batch_count);
    a_ref_array.resize(total_batch_count);
    b_ref_array.resize(total_batch_count);
    c_cast_ref_array.resize(total_batch_count);
    c_ref_array.resize(total_batch_count);

    size_t idx = 0;
    int64_t size_a = 0, size_b = 0, size_c = 0;
    for (int64_t i = 0; i < group_count; i++) {
        switch (layout) {
            case oneapi::math::layout::col_major:
                size_a = lda[i] * ((transa[i] == oneapi::math::transpose::nontrans) ? k[i] : m[i]);
                size_b = ldb[i] * ((transb[i] == oneapi::math::transpose::nontrans) ? n[i] : k[i]);
                size_c = ldc[i] * n[i];
                break;
            case oneapi::math::layout::row_major:
                size_a = lda[i] * ((transa[i] == oneapi::math::transpose::nontrans) ? m[i] : k[i]);
                size_b = ldb[i] * ((transb[i] == oneapi::math::transpose::nontrans) ? k[i] : n[i]);
                size_c = ldc[i] * m[i];
                break;
            default: break;
        }
        for (int64_t j = 0; j < group_size[i]; j++) {
            a_array[idx] = (Ta*)oneapi::math::malloc_shared(64, sizeof(Ta) * size_a, *dev, cxt);
            b_array[idx] = (Tb*)oneapi::math::malloc_shared(64, sizeof(Tb) * size_b, *dev, cxt);
            c_array[idx] = (Tc*)oneapi::math::malloc_shared(64, sizeof(Tc) * size_c, *dev, cxt);
            a_ref_array[idx] = (Ts*)oneapi::math::malloc_shared(64, sizeof(Ts) * size_a, *dev, cxt);
            b_ref_array[idx] = (Ts*)oneapi::math::malloc_shared(64, sizeof(Ts) * size_b, *dev, cxt);
            c_cast_ref_array[idx] =
                (Tc*)oneapi::math::malloc_shared(64, sizeof(Tc) * size_c, *dev, cxt);
            c_ref_array[idx] = (Ts*)oneapi::math::malloc_shared(64, sizeof(Ts) * size_c, *dev, cxt);
            rand_matrix(a_array[idx], layout, transa[i], m[i], k[i], lda[i]);
            rand_matrix(b_array[idx], layout, transb[i], k[i], n[i], ldb[i]);
            rand_matrix(c_array[idx], layout, oneapi::math::transpose::nontrans, m[i], n[i],
                        ldc[i]);
            copy_matrix(a_array[idx], layout, transa[i], m[i], k[i], lda[i], a_ref_array[idx]);
            copy_matrix(b_array[idx], layout, transb[i], k[i], n[i], ldb[i], b_ref_array[idx]);
            copy_matrix(c_array[idx], layout, oneapi::math::transpose::nontrans, m[i], n[i], ldc[i],
                        c_ref_array[idx]);
            idx++;
        }
    }

    // Call reference GEMM_BATCH.
    using fp_ref = typename ref_type_info<Ts>::type;
    int* m_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* n_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* k_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* lda_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* ldb_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* ldc_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);
    int* group_size_ref = (int*)oneapi::math::aligned_alloc(64, sizeof(int) * group_count);

    CBLAS_TRANSPOSE* transa_ref =
        (CBLAS_TRANSPOSE*)oneapi::math::aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);
    CBLAS_TRANSPOSE* transb_ref =
        (CBLAS_TRANSPOSE*)oneapi::math::aligned_alloc(64, sizeof(CBLAS_TRANSPOSE) * group_count);

    if ((m_ref == NULL) || (n_ref == NULL) || (k_ref == NULL) || (lda_ref == NULL) ||
        (ldb_ref == NULL) || (ldc_ref == NULL) || (transa_ref == NULL) || (transb_ref == NULL) ||
        (group_size_ref == NULL)) {
        std::cout << "Error cannot allocate input arrays\n";
        oneapi::math::aligned_free(m_ref);
        oneapi::math::aligned_free(n_ref);
        oneapi::math::aligned_free(k_ref);
        oneapi::math::aligned_free(lda_ref);
        oneapi::math::aligned_free(ldb_ref);
        oneapi::math::aligned_free(ldc_ref);
        oneapi::math::aligned_free(transa_ref);
        oneapi::math::aligned_free(transb_ref);
        oneapi::math::aligned_free(group_size_ref);
        idx = 0;
        for (int64_t i = 0; i < group_count; i++) {
            for (int64_t j = 0; j < group_size[i]; j++) {
                oneapi::math::free_shared(a_array[idx], cxt);
                oneapi::math::free_shared(b_array[idx], cxt);
                oneapi::math::free_shared(c_array[idx], cxt);
                oneapi::math::free_shared(a_ref_array[idx], cxt);
                oneapi::math::free_shared(b_ref_array[idx], cxt);
                oneapi::math::free_shared(c_cast_ref_array[idx], cxt);
                oneapi::math::free_shared(c_ref_array[idx], cxt);
                idx++;
            }
        }
        return false;
    }
    for (unsigned node = 0; node < graph_nodes; node++) {
        idx = 0;
        for (int64_t i = 0; i < group_count; i++) {
            transa_ref[i] = convert_to_cblas_trans(transa[i]);
            transb_ref[i] = convert_to_cblas_trans(transb[i]);
            m_ref[i] = (int)m[i];
            n_ref[i] = (int)n[i];
            k_ref[i] = (int)k[i];
            lda_ref[i] = (int)lda[i];
            ldb_ref[i] = (int)ldb[i];
            ldc_ref[i] = (int)ldc[i];
            group_size_ref[i] = (int)group_size[i];
            for (int64_t j = 0; j < group_size_ref[i]; j++) {
                ::gemm(convert_to_cblas_layout(layout), transa_ref[i], transb_ref[i],
                       (const int*)&m_ref[i], (const int*)&n_ref[i], (const int*)&k_ref[i],
                       (const fp_ref*)&alpha[i], (const fp_ref*)a_ref_array[idx],
                       (const int*)&lda_ref[i], (const fp_ref*)b_ref_array[idx],
                       (const int*)&ldb_ref[i], (const fp_ref*)&beta[i], (fp_ref*)c_ref_array[idx],
                       (const int*)&ldc_ref[i]);
                idx++;
            }
        }
    }

    // Being recording oneMath operations to a graph
    namespace sycl_exp = sycl::ext::oneapi::experimental;
    auto graph = sycl_exp::command_graph(main_queue);
    graph.begin_recording(main_queue);
    try {
        for (unsigned node = 0; node < graph_nodes; node++) {
#ifdef CALL_RT_API
            switch (layout) {
                case oneapi::math::layout::col_major:
                    oneapi::math::blas::column_major::gemm_batch(
                        main_queue, &transa[0], &transb[0], &m[0], &n[0], &k[0], &alpha[0],
                        (const Ta**)&a_array[0], &lda[0], (const Tb**)&b_array[0], &ldb[0],
                        &beta[0], &c_array[0], &ldc[0], group_count, &group_size[0]);
                    break;
                case oneapi::math::layout::row_major:
                    oneapi::math::blas::row_major::gemm_batch(
                        main_queue, &transa[0], &transb[0], &m[0], &n[0], &k[0], &alpha[0],
                        (const Ta**)&a_array[0], &lda[0], (const Tb**)&b_array[0], &ldb[0],
                        &beta[0], &c_array[0], &ldc[0], group_count, &group_size[0]);
                    break;
                default: break;
            }
#else
            switch (layout) {
                case oneapi::math::layout::col_major:
                    TEST_RUN_BLAS_CT_SELECT(
                        main_queue, oneapi::math::blas::column_major::gemm_batch, &transa[0],
                        &transb[0], &m[0], &n[0], &k[0], &alpha[0], (const Ta**)&a_array[0],
                        &lda[0], (const Tb**)&b_array[0], &ldb[0], &beta[0], &c_array[0], &ldc[0],
                        group_count, &group_size[0]);
                    break;
                case oneapi::math::layout::row_major:
                    TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gemm_batch,
                                            &transa[0], &transb[0], &m[0], &n[0], &k[0], &alpha[0],
                                            (const Ta**)&a_array[0], &lda[0],
                                            (const Ta**)&b_array[0], &ldb[0], &beta[0], &c_array[0],
                                            &ldc[0], group_count, &group_size[0]);
                    break;
                default: break;
            }
#endif
        }
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM_BATCH:\n"
                  << e.what() << std::endl;
        print_error_code(e);
    }
    catch (const oneapi::math::unimplemented& e) {
        oneapi::math::aligned_free(m_ref);
        oneapi::math::aligned_free(n_ref);
        oneapi::math::aligned_free(k_ref);
        oneapi::math::aligned_free(lda_ref);
        oneapi::math::aligned_free(ldb_ref);
        oneapi::math::aligned_free(ldc_ref);
        oneapi::math::aligned_free(transa_ref);
        oneapi::math::aligned_free(transb_ref);
        oneapi::math::aligned_free(group_size_ref);
        idx = 0;
        for (int64_t i = 0; i < group_count; i++) {
            for (int64_t j = 0; j < group_size[i]; j++) {
                oneapi::math::free_shared(a_array[idx], cxt);
                oneapi::math::free_shared(b_array[idx], cxt);
                oneapi::math::free_shared(c_array[idx], cxt);
                oneapi::math::free_shared(a_ref_array[idx], cxt);
                oneapi::math::free_shared(b_ref_array[idx], cxt);
                oneapi::math::free_shared(c_cast_ref_array[idx], cxt);
                oneapi::math::free_shared(c_ref_array[idx], cxt);
                idx++;
            }
        }
        return test_skipped;
    }
    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM_BATCH:\n" << error.what() << std::endl;
    }

    // End recording of sycl queue and create executable graph
    graph.end_recording(main_queue);
    auto exec_graph = graph.finalize();

    // Submit graph to execute and wait for completion
    main_queue.ext_oneapi_graph(exec_graph).wait_and_throw();

    bool good = true;
    idx = 0;
    for (int64_t i = 0; i < group_count; i++) {
        const int error_mag = 10 * k[i];
        for (int64_t j = 0; j < group_size[i]; j++) {
            copy_matrix(c_ref_array[idx], layout, oneapi::math::transpose::nontrans, m[i], n[i],
                        ldc[i], c_cast_ref_array[idx]);
            good = good && check_almost_equal_matrix(c_array[idx], c_cast_ref_array[idx], layout,
                                                     m[i], n[i], ldc[i], error_mag, std::cout);
            idx++;
        }
    }
    oneapi::math::aligned_free(m_ref);
    oneapi::math::aligned_free(n_ref);
    oneapi::math::aligned_free(k_ref);
    oneapi::math::aligned_free(lda_ref);
    oneapi::math::aligned_free(ldb_ref);
    oneapi::math::aligned_free(ldc_ref);
    oneapi::math::aligned_free(transa_ref);
    oneapi::math::aligned_free(transb_ref);
    oneapi::math::aligned_free(group_size_ref);
    idx = 0;
    for (int64_t i = 0; i < group_count; i++) {
        for (int64_t j = 0; j < group_size[i]; j++) {
            oneapi::math::free_shared(a_array[idx], cxt);
            oneapi::math::free_shared(b_array[idx], cxt);
            oneapi::math::free_shared(c_array[idx], cxt);
            oneapi::math::free_shared(a_ref_array[idx], cxt);
            oneapi::math::free_shared(b_ref_array[idx], cxt);
            oneapi::math::free_shared(c_cast_ref_array[idx], cxt);
            oneapi::math::free_shared(c_ref_array[idx], cxt);
            idx++;
        }
    }

    return (int)good;
}
#else // ifdef SYCL_EXT_ONEAPI_GRAPH
template <typename Ta, typename Tb, typename Tc, typename Ts>
int test(device*, oneapi::math::layout, int64_t, size_t) {
    // Skip test if graph recording variant and device doesn't support sycl_ext_oneapi_graph
    return 1;
}
#endif

struct GraphGemmBatchUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {
    virtual void SetUp() override {
        // Skip test if graph recording variant and device doesn't support sycl_ext_oneapi_graph
        CHECK_GRAPH_ON_DEVICE(std::get<0>(GetParam()));
    }
};

TEST_P(GraphGemmBatchUsmTests, RealSinglePrecision) {
    sycl::device* dev = std::get<0>(GetParam());
    oneapi::math::layout layout = std::get<1>(GetParam());
    const int64_t group_count = 5;
    const unsigned graph_nodes = 3;
    EXPECT_TRUEORSKIP((test<float, float, float, float>(dev, layout, group_count, graph_nodes)));
}
INSTANTIATE_TEST_SUITE_P(GraphGemmBatchUsmTestSuite, GraphGemmBatchUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
