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

template <typename fp, typename fp_res>
int test(device* dev, oneapi::math::layout layout, int N, int incx) {
    // Prepare data.
    vector<fp> x;
    fp_res result = fp_res(-1), result_ref = fp_res(-1);

    rand_vector(x, N, incx);

    // Call Reference NRM2.
    using fp_ref = typename ref_type_info<fp>::type;
    const int N_ref = N, incx_ref = std::abs(incx);

    result_ref = ::nrm2<fp_ref, fp_res>(&N_ref, (fp_ref*)x.data(), &incx_ref);

    // Call DPC++ NRM2.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during NRM2:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> x_buffer = make_buffer(x);
    buffer<fp_res, 1> result_buffer(&result, range<1>(1));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::math::layout::col_major:
                oneapi::math::blas::column_major::nrm2(main_queue, N, x_buffer, incx,
                                                       result_buffer);
                break;
            case oneapi::math::layout::row_major:
                oneapi::math::blas::row_major::nrm2(main_queue, N, x_buffer, incx, result_buffer);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::math::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::column_major::nrm2, N,
                                        x_buffer, incx, result_buffer);
                break;
            case oneapi::math::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::math::blas::row_major::nrm2, N,
                                        x_buffer, incx, result_buffer);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during NRM2:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::math::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of NRM2:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    auto result_accessor = result_buffer.get_host_access(read_only);
    bool good = check_equal(result_accessor[0], result_ref, N, std::cout);

    return (int)good;
}

class Nrm2Tests : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::math::layout>> {
};

TEST_P(Nrm2Tests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 1)));
    EXPECT_TRUEORSKIP(
        (test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -3)));
}
TEST_P(Nrm2Tests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 2)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, 1)));
    EXPECT_TRUEORSKIP(
        (test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 1357, -3)));
}
TEST_P(Nrm2Tests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(std::get<0>(GetParam()),
                                                        std::get<1>(GetParam()), 1357, 2)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(std::get<0>(GetParam()),
                                                        std::get<1>(GetParam()), 1357, 1)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(std::get<0>(GetParam()),
                                                        std::get<1>(GetParam()), 1357, -3)));
}
TEST_P(Nrm2Tests, ComplexDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(std::get<0>(GetParam()),
                                                          std::get<1>(GetParam()), 1357, 2)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(std::get<0>(GetParam()),
                                                          std::get<1>(GetParam()), 1357, 1)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(std::get<0>(GetParam()),
                                                          std::get<1>(GetParam()), 1357, -3)));
}

INSTANTIATE_TEST_SUITE_P(Nrm2TestSuite, Nrm2Tests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::math::layout::col_major,
                                                            oneapi::math::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
