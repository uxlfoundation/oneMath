/*******************************************************************************
* Copyright 2023 Intel Corporation
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

class Philox4x32x10DeviceSkipAheadTests : public ::testing::TestWithParam<sycl::device*> {};

class Philox4x32x10DeviceSkipAheadExTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10DeviceSkipAheadTests, BinaryPrecision) {
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::philox4x32x10<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::philox4x32x10<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::philox4x32x10<16>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10DeviceSkipAheadExTests, BinaryPrecision) {
    rng_device_test<skip_ahead_ex_test<oneapi::math::rng::device::philox4x32x10<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<skip_ahead_ex_test<oneapi::math::rng::device::philox4x32x10<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<skip_ahead_ex_test<oneapi::math::rng::device::philox4x32x10<16>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10DeviceSkipAheadTestsSuite, Philox4x32x10DeviceSkipAheadTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Philox4x32x10DeviceSkipAheadExTestsSuite,
                         Philox4x32x10DeviceSkipAheadExTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

class Mrg32k3aDeviceSkipAheadTests : public ::testing::TestWithParam<sycl::device*> {};

class Mrg32k3aDeviceSkipAheadExTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mrg32k3aDeviceSkipAheadTests, BinaryPrecision) {
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mrg32k3a<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mrg32k3a<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mrg32k3a<16>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mrg32k3aDeviceSkipAheadExTests, BinaryPrecision) {
    rng_device_test<skip_ahead_ex_test<oneapi::math::rng::device::mrg32k3a<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<skip_ahead_ex_test<oneapi::math::rng::device::mrg32k3a<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<skip_ahead_ex_test<oneapi::math::rng::device::mrg32k3a<16>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mrg32k3aDeviceSkipAheadTestsSuite, Mrg32k3aDeviceSkipAheadTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Mrg32k3aDeviceSkipAheadExTestsSuite, Mrg32k3aDeviceSkipAheadExTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mcg31m1DeviceSkipAheadTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mcg31m1DeviceSkipAheadTests, BinaryPrecision) {
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mcg31m1<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mcg31m1<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mcg31m1<16>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mcg31m1DeviceSkipAheadTestsSuite, Mcg31m1DeviceSkipAheadTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mcg59DeviceSkipAheadTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mcg59DeviceSkipAheadTests, BinaryPrecision) {
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mcg59<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mcg59<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<skip_ahead_test<oneapi::math::rng::device::mcg59<16>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mcg59DeviceSkipAheadTestsSuite, Mcg59DeviceSkipAheadTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // namespace
