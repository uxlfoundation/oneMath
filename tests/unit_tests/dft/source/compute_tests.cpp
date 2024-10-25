/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>

#include "compute_inplace.hpp"
#include "compute_inplace_real_real.hpp"
#include "compute_out_of_place.hpp"
#include "compute_out_of_place_real_real.hpp"

extern std::vector<sycl::device*> devices;

namespace {

class ComputeTests_in_place_COMPLEX
        : public ::testing::TestWithParam<std::tuple<sycl::device*, DFTParams>> {};
class ComputeTests_real_real_in_place_COMPLEX
        : public ::testing::TestWithParam<std::tuple<sycl::device*, DFTParams>> {};
class ComputeTests_out_of_place_COMPLEX
        : public ::testing::TestWithParam<std::tuple<sycl::device*, DFTParams>> {};
class ComputeTests_real_real_out_of_place_COMPLEX
        : public ::testing::TestWithParam<std::tuple<sycl::device*, DFTParams>> {};

class ComputeTests_in_place_REAL
        : public ::testing::TestWithParam<std::tuple<sycl::device*, DFTParams>> {};
class ComputeTests_real_real_in_place_REAL
        : public ::testing::TestWithParam<std::tuple<sycl::device*, DFTParams>> {};
class ComputeTests_out_of_place_REAL
        : public ::testing::TestWithParam<std::tuple<sycl::device*, DFTParams>> {};
class ComputeTests_real_real_out_of_place_REAL
        : public ::testing::TestWithParam<std::tuple<sycl::device*, DFTParams>> {};

#define INSTANTIATE_TEST(PRECISION, DOMAIN, PLACE, LAYOUT, STORAGE)                       \
    TEST_P(ComputeTests##_##LAYOUT##PLACE##_##DOMAIN,                                     \
           DOMAIN##_##PRECISION##_##PLACE##_##LAYOUT##STORAGE) {                          \
        try {                                                                             \
            auto test = DFT_Test<oneapi::mkl::dft::precision::PRECISION,                  \
                                 oneapi::mkl::dft::domain::DOMAIN>{                       \
                std::get<0>(GetParam()), std::get<1>(GetParam()).sizes,                   \
                std::get<1>(GetParam()).strides_fwd, std::get<1>(GetParam()).strides_bwd, \
                std::get<1>(GetParam()).batches                                           \
            };                                                                            \
            EXPECT_TRUEORSKIP(test.test_##PLACE##_##LAYOUT##STORAGE());                   \
        }                                                                                 \
        catch (oneapi::mkl::unimplemented & e) {                                          \
            std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;    \
            GTEST_SKIP();                                                                 \
        }                                                                                 \
        catch (std::exception & e) {                                                      \
            std::string msg = e.what();                                                   \
            if ((msg.find("FFT_UNIMPLEMENTED") != std::string::npos) ||                   \
                (msg.find("Unimplemented") != std::string::npos)) {                       \
                std::cout << "Skipping test because: \"" << msg << "\"" << std::endl;     \
                GTEST_SKIP();                                                             \
            }                                                                             \
            throw;                                                                        \
        }                                                                                 \
    }

#define INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(PLACE, LAYOUT, STORAGE) \
    INSTANTIATE_TEST(SINGLE, COMPLEX, PLACE, LAYOUT, STORAGE)                \
    INSTANTIATE_TEST(SINGLE, REAL, PLACE, LAYOUT, STORAGE)                   \
    INSTANTIATE_TEST(DOUBLE, COMPLEX, PLACE, LAYOUT, STORAGE)                \
    INSTANTIATE_TEST(DOUBLE, REAL, PLACE, LAYOUT, STORAGE)

#define INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN_PLACE_LAYOUT(STORAGE)      \
    INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(in_place, , STORAGE)           \
    INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(in_place, real_real_, STORAGE) \
    INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(out_of_place, , STORAGE)       \
    INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN(out_of_place, real_real_, STORAGE)

INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN_PLACE_LAYOUT(buffer)
INSTANTIATE_TEST_DIMENSIONS_PRECISION_DOMAIN_PLACE_LAYOUT(USM)

using shape = std::vector<std::int64_t>;
using i64 = std::int64_t;
// Parameter format - { shape of transform, number of transforms } or { shape, forward strides, backward strides, number of transforms }
// strides need to be chosen in a way that also makes sense for real transforms
std::vector<DFTParams> test_params{
    { shape{ 8 }, i64{ 1 } },
    { shape{ 9 }, i64{ 2 } },
    { shape{ 8 }, i64{ 27 } },
    { shape{ 22 }, i64{ 1 } },
    { shape{ 128 }, i64{ 1 } },

    { shape{ 4, 4 }, i64{ 1 } },
    { shape{ 4, 4 }, i64{ 2 } },
    { shape{ 4, 3 }, i64{ 9 } },
    { shape{ 7, 8 }, i64{ 1 } },
    { shape{ 64, 5 }, i64{ 1 } },

    { shape{ 2, 2, 2 }, i64{ 1 } },
    { shape{ 2, 2, 3 }, i64{ 2 } },
    { shape{ 2, 2, 2 }, i64{ 27 } },
    { shape{ 3, 7, 2 }, i64{ 1 } },
    { shape{ 8, 8, 9 }, i64{ 1 } },

    { shape{ 4, 3 }, shape{ 2, 3, 1 }, shape{ 2, 3, 1 }, i64{ 2 } },
    { shape{ 4, 3 }, shape{ 0, 4, 1 }, shape{ 0, 3, 1 }, i64{ 3 } },
    { shape{ 4, 3 }, shape{ 4, 6, 2 }, shape{ 2, 6, 2 }, i64{ 2 } },
    { shape{ 4, 3 }, shape{ 1, 1, 4 }, shape{ 1, 1, 4 }, i64{ 9 } },
    { shape{ 4, 4 }, shape{ 2, 4, 1 }, shape{ 0, 4, 1 }, i64{ 2 } },
    { shape{ 4, 4 }, shape{ 0, 1, 5 }, shape{ 0, 1, 4 }, i64{ 2 } },
    { shape{ 4, 4 }, shape{ 0, 1, 4 }, shape{ 0, 2, 9 }, i64{ 2 } },
    { shape{ 4, 4 }, shape{ 0, 7, 1 }, shape{ 0, 5, 1 }, i64{ 2 } },
    { shape{ 4, 4 }, shape{ 0, 8, 2 }, shape{ 0, 8, 2 }, i64{ 2 } },
    { shape{ 4, 4 }, shape{ 0, 4, 1 }, shape{ 0, 1, 4 }, i64{ 2 } },

    { shape{ 4, 4, 4 }, shape{ 2, 1, 4, 16 }, shape{ 4, 1, 4, 16 }, i64{ 2 } },
    { shape{ 4, 4, 4 }, shape{ 4, 17, 4, 1 }, shape{ 4, 23, 5, 1 }, i64{ 2 } },
    { shape{ 4, 4, 4 }, shape{ 0, 32, 8, 2 }, shape{ 0, 32, 8, 2 }, i64{ 2 } },
    { shape{ 4, 4, 4 }, shape{ 2, 4, 1, 16 }, shape{ 1, 4, 16, 1 }, i64{ 2 } },
    { shape{ 4, 4, 4 }, shape{ 0, 1, 32, 8 }, shape{ 0, 1, 32, 8 }, i64{ 2 } },
};
std::vector<DFTParams> test_params_real_in_place{
    { shape{ 8 }, i64{ 1 } },
    { shape{ 9 }, i64{ 2 } },
    { shape{ 8 }, i64{ 27 } },
    { shape{ 22 }, i64{ 1 } },
    { shape{ 128 }, i64{ 1 } },

    { shape{ 4, 4 }, i64{ 1 } },
    { shape{ 4, 4 }, i64{ 2 } },
    { shape{ 4, 3 }, i64{ 9 } },
    { shape{ 7, 8 }, i64{ 1 } },
    { shape{ 64, 5 }, i64{ 1 } },

    { shape{ 2, 2, 2 }, i64{ 1 } },
    { shape{ 2, 2, 3 }, i64{ 2 } },
    { shape{ 2, 2, 2 }, i64{ 27 } },
    { shape{ 3, 7, 2 }, i64{ 1 } },
    { shape{ 8, 8, 9 }, i64{ 1 } },

    { shape{ 4, 3 }, shape{ 0, 4, 1 }, shape{ 0, 2, 1 }, i64{ 2 } },
    { shape{ 4, 3 }, shape{ 0, 6, 1 }, shape{ 0, 3, 1 }, i64{ 2 } },
    { shape{ 4, 3 }, shape{ 0, 8, 2 }, shape{ 0, 4, 2 }, i64{ 2 } },
    { shape{ 4, 3 }, shape{ 2, 4, 1 }, shape{ 1, 2, 1 }, i64{ 2 } },
    { shape{ 4, 3 }, shape{ 6, 1, 4 }, shape{ 3, 1, 4 }, i64{ 9 } },
    { shape{ 4, 3 }, shape{ 0, 1, 5 }, shape{ 0, 1, 5 }, i64{ 2 } },
    { shape{ 4, 3 }, shape{ 0, 3, 12 }, shape{ 0, 3, 12 }, i64{ 9 } },

    { shape{ 4, 4, 4 }, shape{ 4, 1, 4, 16 }, shape{ 2, 1, 4, 16 }, i64{ 2 } },
    { shape{ 4, 4, 4 }, shape{ 0, 48, 12, 2 }, shape{ 0, 24, 6, 2 }, i64{ 2 } },
    { shape{ 4, 4, 4 }, shape{ 0, 1, 48, 8 }, shape{ 0, 1, 24, 8 }, i64{ 2 } },
};

INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests_in_place_COMPLEX,
                         testing::Combine(testing::ValuesIn(devices),
                                          testing::ValuesIn(test_params)),
                         DFTParamsPrint{});
INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests_real_real_in_place_COMPLEX,
                         testing::Combine(testing::ValuesIn(devices),
                                          testing::ValuesIn(test_params)),
                         DFTParamsPrint{});
INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests_out_of_place_COMPLEX,
                         testing::Combine(testing::ValuesIn(devices),
                                          testing::ValuesIn(test_params)),
                         DFTParamsPrint{});
INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests_real_real_out_of_place_COMPLEX,
                         testing::Combine(testing::ValuesIn(devices),
                                          testing::ValuesIn(test_params)),
                         DFTParamsPrint{});

INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests_in_place_REAL,
                         testing::Combine(testing::ValuesIn(devices),
                                          testing::ValuesIn(test_params_real_in_place)),
                         DFTParamsPrint{});
INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests_real_real_in_place_REAL,
                         testing::Combine(testing::ValuesIn(devices),
                                          testing::ValuesIn(test_params_real_in_place)),
                         DFTParamsPrint{});
INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests_out_of_place_REAL,
                         testing::Combine(testing::ValuesIn(devices),
                                          testing::ValuesIn(test_params)),
                         DFTParamsPrint{});
INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests_real_real_out_of_place_REAL,
                         testing::Combine(testing::ValuesIn(devices),
                                          testing::ValuesIn(test_params)),
                         DFTParamsPrint{});

} // anonymous namespace
