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

template <typename fp, typename fp_scalar>
int test(device* dev, oneapi::math::layout layout) {
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

    buffer<fp, 1> a_buffer(&a, range<1>(1));
    buffer<fp, 1> b_buffer(&b, range<1>(1));
    buffer<fp_scalar, 1> c_buffer(&c, range<1>(1));
    buffer<fp, 1> s_buffer(&s, range<1>(1));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::rotg(main_queue, a_buffer, b_buffer, c_buffer,
                                                       s_buffer);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::rotg(main_queue, a_buffer, b_buffer, c_buffer,
                                                    s_buffer);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::rotg,
                                        a_buffer, b_buffer, c_buffer, s_buffer);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::rotg, a_buffer,
                                        b_buffer, c_buffer, s_buffer);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during ROTG:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of ROTG:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto a_accessor = a_buffer.get_host_access(read_only);
    bool good_a = check_equal(a_accessor[0], a_ref, 4, std::cout);
    auto b_accessor = b_buffer.get_host_access(read_only);
    bool good_b = check_equal(b_accessor[0], b_ref, 4, std::cout);
    auto s_accessor = s_buffer.get_host_access(read_only);
    bool good_s = check_equal(s_accessor[0], s_ref, 4, std::cout);
    auto c_accessor = c_buffer.get_host_access(read_only);
    bool good_c = check_equal(c_accessor[0], c_ref, 4, std::cout);

    bool good = good_a && good_b && good_c && good_s;

    return (int)good;
}

class RotgTests : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {
};

TEST_P(RotgTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP((test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP((test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP((test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotgTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP((test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP((test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP((test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotgTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotgTests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}

INSTANTIATE_TEST_SUITE_P(RotgTestSuite, RotgTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
