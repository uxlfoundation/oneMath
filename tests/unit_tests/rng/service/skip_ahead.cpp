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

#include "skip_ahead_test.hpp"

#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

namespace {

class Philox4x32x10SkipAheadTests : public ::testing::TestWithParam<sycl::device*> {};

class Philox4x32x10SkipAheadExTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10SkipAheadTests, BinaryPrecision) {
    rng_test<skip_ahead_test<oneapi::math::rng::philox4x32x10>> test;
    EXPECT_TRUEORSKIP((test(GetParam())));
}

TEST_P(Philox4x32x10SkipAheadExTests, BinaryPrecision) {
    rng_test<skip_ahead_ex_test<oneapi::math::rng::philox4x32x10>> test;
    EXPECT_TRUEORSKIP((test(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10SkipAheadTestSuite, Philox4x32x10SkipAheadTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Philox4x32x10SkipAheadExTestSuite, Philox4x32x10SkipAheadExTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mrg32k3aSkipAheadTests : public ::testing::TestWithParam<sycl::device*> {};

class Mrg32k3aSkipAheadExTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mrg32k3aSkipAheadTests, BinaryPrecision) {
    rng_test<skip_ahead_test<oneapi::math::rng::mrg32k3a>> test;
    EXPECT_TRUEORSKIP((test(GetParam())));
}

TEST_P(Mrg32k3aSkipAheadExTests, BinaryPrecision) {
    rng_test<skip_ahead_ex_test<oneapi::math::rng::mrg32k3a>> test;
    EXPECT_TRUEORSKIP((test(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mrg32k3aSkipAheadTestSuite, Mrg32k3aSkipAheadTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Mrg32k3aSkipAheadExTestSuite, Mrg32k3aSkipAheadExTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
