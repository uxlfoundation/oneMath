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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
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

template <typename fp, typename fp_scalar, usm::alloc alloc_type = usm::alloc::shared>
int test(device* dev, oneapi::mkl::layout layout) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during ROTG:\n"
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

    fp a, b, s, a_ref, b_ref, s_ref;
    fp_scalar c, c_ref;

    a = rand_scalar<fp>();
    b = rand_scalar<fp>();
    s = rand_scalar<fp>();
    c = rand_scalar<fp_scalar>();
    a_ref = a;
    b_ref = b;
    s_ref = s;
    c_ref = c;

    // Call Reference ROTG.
    using fp_ref = typename ref_type_info<fp>::type;

    ::rotg((fp_ref*)&a_ref, (fp_ref*)&b_ref, (fp_scalar*)&c_ref, (fp_ref*)&s_ref);

    // Call DPC++ ROTG.
    fp *a_p, *b_p, *s_p;
    fp_scalar* c_p;
    if constexpr (alloc_type == usm::alloc::shared) {
        a_p = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
        b_p = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
        s_p = (fp*)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
        c_p = (fp_scalar*)oneapi::mkl::malloc_shared(64, sizeof(fp_scalar), *dev, cxt);
    }
    else if constexpr (alloc_type == usm::alloc::device) {
        a_p = (fp*)oneapi::mkl::malloc_device(64, sizeof(fp), *dev, cxt);
        b_p = (fp*)oneapi::mkl::malloc_device(64, sizeof(fp), *dev, cxt);
        s_p = (fp*)oneapi::mkl::malloc_device(64, sizeof(fp), *dev, cxt);
        c_p = (fp_scalar*)oneapi::mkl::malloc_device(64, sizeof(fp_scalar), *dev, cxt);
    }
    else {
        throw std::runtime_error("Bad alloc_type");
    }

    main_queue.memcpy(a_p, &a, sizeof(fp));
    main_queue.memcpy(b_p, &b, sizeof(fp));
    main_queue.memcpy(s_p, &s, sizeof(fp));
    main_queue.memcpy(c_p, &c, sizeof(fp_scalar));
    main_queue.wait();

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                done = oneapi::mkl::blas::column_major::rotg(main_queue, a_p, b_p, c_p, s_p,
                                                             dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::rotg(main_queue, a_p, b_p, c_p, s_p,
                                                          dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::rotg, a_p, b_p,
                                        c_p, s_p, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::rotg, a_p, b_p,
                                        c_p, s_p, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during ROTG:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of ROTG:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good_a = check_equal_ptr(main_queue, a_p, a_ref, 4, std::cout);
    bool good_b = check_equal_ptr(main_queue, b_p, b_ref, 4, std::cout);
    bool good_s = check_equal_ptr(main_queue, s_p, s_ref, 4, std::cout);
    bool good_c = check_equal_ptr(main_queue, c_p, c_ref, 4, std::cout);

    bool good = good_a && good_b && good_c && good_s;

    oneapi::mkl::free_usm(a_p, cxt);
    oneapi::mkl::free_usm(b_p, cxt);
    oneapi::mkl::free_usm(s_p, cxt);
    oneapi::mkl::free_usm(c_p, cxt);

    return (int)good;
}

class RotgUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {};

TEST_P(RotgUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP((test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP(
        (test<float, float, usm::alloc::device>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotgUsmTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP((test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP((test<double, double, usm::alloc::device>(std::get<0>(GetParam()),
                                                                std::get<1>(GetParam()))));
}
TEST_P(RotgUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float, usm::alloc::device>(
        std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotgUsmTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double, usm::alloc::device>(
        std::get<0>(GetParam()), std::get<1>(GetParam()))));
}

INSTANTIATE_TEST_SUITE_P(RotgUsmTestSuite, RotgUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
