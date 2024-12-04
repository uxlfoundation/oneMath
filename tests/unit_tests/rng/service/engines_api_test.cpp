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

#include "engines_api_tests.hpp"

#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

namespace {

class Philox4x32x10ConstructorsTests : public ::testing::TestWithParam<sycl::device*> {};

class Philox4x32x10CopyTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10ConstructorsTests, BinaryPrecision) {
    rng_test<engines_constructors_test<oneapi::math::rng::philox4x32x10>> test;
    std::initializer_list<std::uint64_t> seed_ex = { SEED, 0, 0 };
    EXPECT_TRUEORSKIP((test(GetParam(), seed_ex)));
}

TEST_P(Philox4x32x10CopyTests, BinaryPrecision) {
    rng_test<engines_copy_test<oneapi::math::rng::philox4x32x10>> test;
    EXPECT_TRUEORSKIP((test(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10ConstructorsTestsuite, Philox4x32x10ConstructorsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Philox4x32x10CopyTestsuite, Philox4x32x10CopyTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mrg32k3aConstructorsTests : public ::testing::TestWithParam<sycl::device*> {};

class Mrg32k3aCopyTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mrg32k3aConstructorsTests, BinaryPrecision) {
    rng_test<engines_constructors_test<oneapi::math::rng::mrg32k3a>> test;
    std::initializer_list<std::uint32_t> seed_ex = { SEED, 1, 1, 1, 1, 1 };
    EXPECT_TRUEORSKIP((test(GetParam(), seed_ex)));
}

TEST_P(Mrg32k3aCopyTests, BinaryPrecision) {
    rng_test<engines_copy_test<oneapi::math::rng::mrg32k3a>> test;
    EXPECT_TRUEORSKIP((test(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mrg32k3aConstructorsTestsuite, Mrg32k3aConstructorsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Mrg32k3aCopyTestsuite, Mrg32k3aCopyTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
