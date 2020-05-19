/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <CL/sycl.hpp>
#include "cblas.h"
#include "onemkl/detail/config.hpp"
#include "onemkl/onemkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"
#include "usm_allocator_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp, typename fp_scalar>
int test(const device &dev, int N, int incx, int incy, fp_scalar c, fp_scalar s) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during ROT:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator_helper<fp, 64>(cxt, dev);
    vector<fp, decltype(ua)> x(ua), x_ref(ua), y(ua), y_ref(ua);
    rand_vector(x, N, incx);
    rand_vector(y, N, incy);

    x_ref.resize(x.size());
    for (int i = 0; i < x.size(); i++)
        x_ref[i] = x[i];

    y_ref.resize(y.size());
    for (int i = 0; i < y.size(); i++)
        y_ref[i] = y[i];

    // Call Reference ROT.
    using fp_ref    = typename ref_type_info<fp>::type;
    const int N_ref = N, incx_ref = incx, incy_ref = incy;

    ::rot(&N_ref, (fp_ref *)x_ref.data(), &incx_ref, (fp_ref *)y_ref.data(), &incy_ref,
          (fp_scalar *)&c, (fp_scalar *)&s);

    // Call DPC++ ROT.

    try {
#ifdef CALL_RT_API
        done = onemkl::blas::rot(main_queue, N, x.data(), incx, y.data(), incy, c, s, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, onemkl::blas::rot,
                    (main_queue, N, x.data(), incx, y.data(), incy, c, s, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during ROT:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const backend_unsupported_exception &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of ROT_USM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good;
    {
        bool good_x = check_equal_vector(x, x_ref, N, incx, N, std::cout);
        bool good_y = check_equal_vector(y, y_ref, N, incy, N, std::cout);
        good        = good_x && good_y;
    }

    return (int)good;
}

class RotUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(RotUsmTests, RealSinglePrecision) {
    float c(2.0);
    float s(-0.5);
    EXPECT_TRUEORSKIP((test<float, float>(GetParam(), 1357, 2, 3, c, s)));
    EXPECT_TRUEORSKIP((test<float, float>(GetParam(), 1357, 1, 1, c, s)));
    EXPECT_TRUEORSKIP((test<float, float>(GetParam(), 1357, -2, -3, c, s)));
}
TEST_P(RotUsmTests, RealDoublePrecision) {
    double c(2.0);
    double s(-0.5);
    EXPECT_TRUEORSKIP((test<double, double>(GetParam(), 1357, 2, 3, c, s)));
    EXPECT_TRUEORSKIP((test<double, double>(GetParam(), 1357, 1, 1, c, s)));
    EXPECT_TRUEORSKIP((test<double, double>(GetParam(), 1357, -2, -3, c, s)));
}
TEST_P(RotUsmTests, ComplexSinglePrecision) {
    float c = 2.0;
    float s = -0.5;
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), 1357, 2, 3, c, s)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), 1357, 1, 1, c, s)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), 1357, -2, -3, c, s)));
}
TEST_P(RotUsmTests, ComplexDoublePrecision) {
    double c = 2.0;
    double s = -0.5;
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), 1357, 2, 3, c, s)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), 1357, 1, 1, c, s)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), 1357, -2, -3, c, s)));
}

INSTANTIATE_TEST_SUITE_P(RotUsmTestSuite, RotUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
