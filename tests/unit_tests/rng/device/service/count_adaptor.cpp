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

#include "count_adaptor_test.hpp"

#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

namespace {

class Philox4x32x10DeviceCountAdaptorTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10DeviceCountAdaptorTests, BinaryPrecision) {
    rng_device_test<count_adaptor_test<oneapi::math::rng::device::philox4x32x10<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<count_adaptor_test<oneapi::math::rng::device::philox4x32x10<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10DeviceCountAdaptorTestsSuite,
                         Philox4x32x10DeviceCountAdaptorTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

class Mrg32k3aDeviceCountAdaptorTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mrg32k3aDeviceCountAdaptorTests, BinaryPrecision) {
    rng_device_test<count_adaptor_test<oneapi::math::rng::device::mrg32k3a<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<count_adaptor_test<oneapi::math::rng::device::mrg32k3a<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mrg32k3aDeviceCountAdaptorTestsSuite, Mrg32k3aDeviceCountAdaptorTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mcg31m1DeviceCountAdaptorTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mcg31m1DeviceCountAdaptorTests, BinaryPrecision) {
    rng_device_test<count_adaptor_test<oneapi::math::rng::device::mcg31m1<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<count_adaptor_test<oneapi::math::rng::device::mcg31m1<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mcg31m1DeviceCountAdaptorTestsSuite, Mcg31m1DeviceCountAdaptorTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mcg59DeviceCountAdaptorTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mcg59DeviceCountAdaptorTests, BinaryPrecision) {
    rng_device_test<count_adaptor_test<oneapi::math::rng::device::mcg59<1>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<count_adaptor_test<oneapi::math::rng::device::mcg59<4>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mcg59DeviceCountAdaptorTestsSuite, Mcg59DeviceCountAdaptorTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // namespace
