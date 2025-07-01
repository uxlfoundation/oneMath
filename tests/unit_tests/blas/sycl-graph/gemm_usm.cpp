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
#include "onemath_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;

extern std::vector<sycl::device*> devices;

namespace {
#ifdef SYCL_EXT_ONEAPI_GRAPH
template <typename Ta, typename Tc>
int test(device* dev, oneapi::math::layout layout, int m, int n, int k, int lda, int ldb, int ldc,
         Tc alpha, Tc beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    // Test with 4 nodes in the graph, for each combination of matrix A & B being transposed
    // or not.
    constexpr size_t num_ops = 4;
    std::array<std::pair<oneapi::math::transpose, oneapi::math::transpose>, num_ops> trans = {
        std::make_pair(oneapi::math::transpose::nontrans, oneapi::math::transpose::nontrans),
        std::make_pair(oneapi::math::transpose::nontrans, oneapi::math::transpose::trans),
        std::make_pair(oneapi::math::transpose::trans, oneapi::math::transpose::nontrans),
        std::make_pair(oneapi::math::transpose::trans, oneapi::math::transpose::trans),
    };

    queue main_queue(*dev, exception_handler, property::queue::in_order{});

    // Prepare data. Have a single transposed and non-transposed matrix allocation that's
    // reused across all nodes in the graph.
    auto ua = usm_allocator<Ta, usm::alloc::shared, 64>(main_queue);
    auto uc = usm_allocator<Tc, usm::alloc::shared, 64>(main_queue);
    std::vector<Ta, decltype(ua)> Trans(ua), NoTrans(ua);
    rand_matrix(Trans, layout, oneapi::math::transpose::trans, m, k, lda);
    rand_matrix(NoTrans, layout, oneapi::math::transpose::nontrans, m, k, lda);

    // Created input/output matrix C that is a data dependency across nodes, and
    // C_ref used to verify it's final value against the cblas reference.
    std::vector<Tc, decltype(uc)> C(ua);
    rand_matrix(C, layout, oneapi::math::transpose::nontrans, m, n, ldc);
    auto C_ref = C;

    // Being recording oneMath operations to a graph
    namespace sycl_exp = sycl::ext::oneapi::experimental;
    auto graph = sycl_exp::command_graph(main_queue);
    graph.begin_recording(main_queue);
    for (auto [transa, transb] : trans) {
        // Assign transpose or non-transpose matrix to A & B
        Ta* A = transa == oneapi::math::transpose::trans ? Trans.data() : NoTrans.data();
        Ta* B = transa == oneapi::math::transpose::trans ? Trans.data() : NoTrans.data();

        // Calculate reference
        const int m_ref = m, n_ref = n, k_ref = k;
        const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

        using ta_ref = typename ref_type_info<Ta>::type;
        using tc_ref = typename ref_type_info<Tc>::type;
        ::gemm(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa),
               convert_to_cblas_trans(transb), &m_ref, &n_ref, &k_ref, (tc_ref*)&alpha, (ta_ref*)A,
               &lda_ref, (ta_ref*)B, &ldb_ref, (tc_ref*)&beta, (tc_ref*)C_ref.data(), &ldc_ref);

        // Submit oneMath gemm operation to main_queue in recording mode
        try {
#ifdef CALL_RT_API
            switch (layout) {
                case oneapi::math::layout::col_major:
                    oneapi::math::blas::column_major::gemm(main_queue, transa, transb, m, n, k,
                                                           alpha, A, lda, B, ldb, beta, C.data(),
                                                           ldc);
                    break;
                case oneapi::math::layout::row_major:
                    oneapi::math::blas::row_major::gemm(main_queue, transa, transb, m, n, k, alpha,
                                                        A, lda, B, ldb, beta, C.data(), ldc);
                    break;
                default: break;
            }
#else
            switch (layout) {
                case oneapi::math::layout::col_major:
                    TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::gemm,
                                            transa, transb, m, n, k, alpha, A, lda, B, ldb, beta,
                                            C.data(), ldc);
                    break;
                case oneapi::math::layout::row_major:
                    TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::gemm, transa,
                                            transb, m, n, k, alpha, A, lda, B, ldb, beta, C.data(),
                                            ldc);
                    break;
                default: break;
            }
#endif
        }
        catch (exception const& e) {
            std::cout << "Caught synchronous SYCL exception during GEMM:\n"
                      << e.what() << std::endl;
            print_error_code(e);
        }
        catch (const oneapi::math::unimplemented& e) {
            return test_skipped;
        }
        catch (const std::runtime_error& error) {
            std::cout << "Error raised during execution of GEMM:\n" << error.what() << std::endl;
        }
    }
    // End recording of sycl queue and create executable graph
    graph.end_recording(main_queue);
    auto exec_graph = graph.finalize();

    // Submit graph to execute and wait for completion
    main_queue.ext_oneapi_graph(exec_graph).wait_and_throw();

    // Verify graph output against reference
    bool good = check_equal_matrix(C, C_ref, layout, m, n, ldc, 10 * k, std::cout);
    return (int)good;
}

#else // ifdef SYCL_EXT_ONEAPI_GRAPH
template <typename Ta, typename Tc>
int test(device*, oneapi::math::layout, int, int, int, int, int, int, Tc, Tc) {
    // Stub test for SYCL compilers that don't define the sycl_ext_oneapi_graph extension
    return 1;
}
#endif

struct GraphGemmUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {
    virtual void SetUp() override {
        auto device = std::get<0>(GetParam());
        if (device->get_backend() == sycl::backend::opencl) {
            GTEST_SKIP() << "Test not yet supported on OpenCL";
        }
#if SYCL_EXT_ONEAPI_BACKEND_HIP
        if (device->get_backend() == sycl::backend::ext_oneapi_hip) {
            GTEST_SKIP() << "Test not yet supported on HIP";
        }
#endif

        // Skip test if graph recording variant and device doesn't support sycl_ext_oneapi_graph
        CHECK_GRAPH_ON_DEVICE(device);
    }
};

TEST_P(GraphGemmUsmTests, RealSinglePrecision) {
    device* dev = std::get<0>(GetParam());
    oneapi::math::layout layout = std::get<1>(GetParam());

    const int m(1);
    const int n(2);
    const int k(3);
    const int lda(4);
    const int ldb(5);
    const int ldc(6);
    const float alpha(2.0);
    const float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, float>(dev, layout, m, n, k, lda, ldb, ldc, alpha, beta)));
}

INSTANTIATE_TEST_SUITE_P(GraphGemmUsmTestSuite, GraphGemmUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
