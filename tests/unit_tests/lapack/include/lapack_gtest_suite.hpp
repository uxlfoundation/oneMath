/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

using RealSinglePrecisionBuffer = sycl::buffer<float, 1>;
using RealDoublePrecisionBuffer = sycl::buffer<double, 1>;
using ComplexSinglePrecisionBuffer = sycl::buffer<std::complex<float>, 1>;
using ComplexDoublePrecisionBuffer = sycl::buffer<std::complex<double>, 1>;
using RealSinglePrecisionUsm = float;
using RealDoublePrecisionUsm = double;
using ComplexSinglePrecisionUsm = std::complex<float>;
using ComplexDoublePrecisionUsm = std::complex<double>;

#define CREATE_TEST_CLASS(SUITE, TEST) \
    class SUITE##TEST : public ::testing::TestWithParam<sycl::device*> {}

#define INSTANTIATE_TEST_CLASS(SUITE, TEST) \
    INSTANTIATE_TEST_SUITE_P(SUITE, SUITE##TEST, ::testing::ValuesIn(devices), DeviceNamePrint());

#define INSTANTIATE_GTEST_SUITE_ACCURACY(SUITE) \
    CREATE_TEST_CLASS(SUITE, AccuracyUsm);      \
    DEFINE_TEST_ACCURACY_USM_REAL(SUITE);       \
    DEFINE_TEST_ACCURACY_USM_COMPLEX(SUITE);    \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyUsm); \
    CREATE_TEST_CLASS(SUITE, AccuracyBuffer);   \
    DEFINE_TEST_ACCURACY_BUFFER_REAL(SUITE);    \
    DEFINE_TEST_ACCURACY_BUFFER_COMPLEX(SUITE); \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyBuffer)

#define INSTANTIATE_GTEST_SUITE_ACCURACY_REAL(SUITE) \
    CREATE_TEST_CLASS(SUITE, AccuracyUsm);           \
    DEFINE_TEST_ACCURACY_USM_REAL(SUITE);            \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyUsm);      \
    CREATE_TEST_CLASS(SUITE, AccuracyBuffer);        \
    DEFINE_TEST_ACCURACY_BUFFER_REAL(SUITE);         \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyBuffer)

#define INSTANTIATE_GTEST_SUITE_ACCURACY_COMPLEX(SUITE) \
    CREATE_TEST_CLASS(SUITE, AccuracyUsm);              \
    DEFINE_TEST_ACCURACY_USM_COMPLEX(SUITE);            \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyUsm);         \
    CREATE_TEST_CLASS(SUITE, AccuracyBuffer);           \
    DEFINE_TEST_ACCURACY_BUFFER_COMPLEX(SUITE);         \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyBuffer)

#define INSTANTIATE_GTEST_SUITE_ACCURACY_USM(SUITE) \
    CREATE_TEST_CLASS(SUITE, AccuracyUsm);          \
    DEFINE_TEST_ACCURACY_USM_REAL(SUITE);           \
    DEFINE_TEST_ACCURACY_USM_COMPLEX(SUITE);        \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyUsm)

#define INSTANTIATE_GTEST_SUITE_ACCURACY_USM_REAL(SUITE) \
    CREATE_TEST_CLASS(SUITE, AccuracyUsm);               \
    DEFINE_TEST_ACCURACY_USM_REAL(SUITE);                \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyUsm)

#define INSTANTIATE_GTEST_SUITE_ACCURACY_USM_COMPLEX(SUITE) \
    CREATE_TEST_CLASS(SUITE, AccuracyUsm);                  \
    DEFINE_TEST_ACCURACY_USM_COMPLEX(SUITE);                \
    INSTANTIATE_TEST_CLASS(SUITE, AccuracyUsm)

#define DEFINE_TEST_ACCURACY_USM_REAL(SUITE)                                           \
    TEST_P(SUITE##AccuracyUsm, RealSinglePrecision) {                                  \
        test_log::padding = "[          ] ";                                           \
        EXPECT_TRUEORSKIP(                                                             \
            accuracy_controller.run(::accuracy<RealSinglePrecisionUsm>, *GetParam())); \
    }                                                                                  \
    TEST_P(SUITE##AccuracyUsm, RealDoublePrecision) {                                  \
        CHECK_DOUBLE_ON_DEVICE(GetParam());                                            \
        test_log::padding = "[          ] ";                                           \
        EXPECT_TRUEORSKIP(                                                             \
            accuracy_controller.run(::accuracy<RealDoublePrecisionUsm>, *GetParam())); \
    }

#define DEFINE_TEST_ACCURACY_USM_COMPLEX(SUITE)                                           \
    TEST_P(SUITE##AccuracyUsm, ComplexSinglePrecision) {                                  \
        test_log::padding = "[          ] ";                                              \
        EXPECT_TRUEORSKIP(                                                                \
            accuracy_controller.run(::accuracy<ComplexSinglePrecisionUsm>, *GetParam())); \
    }                                                                                     \
    TEST_P(SUITE##AccuracyUsm, ComplexDoublePrecision) {                                  \
        CHECK_DOUBLE_ON_DEVICE(GetParam());                                               \
        test_log::padding = "[          ] ";                                              \
        EXPECT_TRUEORSKIP(                                                                \
            accuracy_controller.run(::accuracy<ComplexDoublePrecisionUsm>, *GetParam())); \
    }

#define DEFINE_TEST_ACCURACY_BUFFER_REAL(SUITE)                                           \
    TEST_P(SUITE##AccuracyBuffer, RealSinglePrecision) {                                  \
        test_log::padding = "[          ] ";                                              \
        EXPECT_TRUEORSKIP(                                                                \
            accuracy_controller.run(::accuracy<RealSinglePrecisionBuffer>, *GetParam())); \
    }                                                                                     \
    TEST_P(SUITE##AccuracyBuffer, RealDoublePrecision) {                                  \
        CHECK_DOUBLE_ON_DEVICE(GetParam());                                               \
        test_log::padding = "[          ] ";                                              \
        EXPECT_TRUEORSKIP(                                                                \
            accuracy_controller.run(::accuracy<RealDoublePrecisionBuffer>, *GetParam())); \
    }

#define DEFINE_TEST_ACCURACY_BUFFER_COMPLEX(SUITE)                                           \
    TEST_P(SUITE##AccuracyBuffer, ComplexSinglePrecision) {                                  \
        test_log::padding = "[          ] ";                                                 \
        EXPECT_TRUEORSKIP(                                                                   \
            accuracy_controller.run(::accuracy<ComplexSinglePrecisionBuffer>, *GetParam())); \
    }                                                                                        \
    TEST_P(SUITE##AccuracyBuffer, ComplexDoublePrecision) {                                  \
        CHECK_DOUBLE_ON_DEVICE(GetParam());                                                  \
        test_log::padding = "[          ] ";                                                 \
        EXPECT_TRUEORSKIP(                                                                   \
            accuracy_controller.run(::accuracy<ComplexDoublePrecisionBuffer>, *GetParam())); \
    }

#define INSTANTIATE_GTEST_SUITE_DEPENDENCY(SUITE) \
    CREATE_TEST_CLASS(SUITE, DependencyUsm);      \
    DEFINE_TEST_DEPENDENCY_REAL(SUITE);           \
    DEFINE_TEST_DEPENDENCY_COMPLEX(SUITE);        \
    INSTANTIATE_TEST_CLASS(SUITE, DependencyUsm)

#define INSTANTIATE_GTEST_SUITE_DEPENDENCY_REAL(SUITE) \
    CREATE_TEST_CLASS(SUITE, DependencyUsm);           \
    DEFINE_TEST_DEPENDENCY_REAL(SUITE);                \
    INSTANTIATE_TEST_CLASS(SUITE, DependencyUsm);

#define INSTANTIATE_GTEST_SUITE_DEPENDENCY_COMPLEX(SUITE) \
    CREATE_TEST_CLASS(SUITE, DependencyUsm);              \
    DEFINE_TEST_DEPENDENCY_COMPLEX(SUITE);                \
    INSTANTIATE_TEST_CLASS(SUITE, DependencyUsm);

#define DEFINE_TEST_DEPENDENCY_REAL(SUITE)                                                     \
    TEST_P(SUITE##DependencyUsm, RealSinglePrecision) {                                        \
        test_log::padding = "[          ] ";                                                   \
        EXPECT_TRUEORSKIP(                                                                     \
            dependency_controller.run(::usm_dependency<RealSinglePrecisionUsm>, *GetParam())); \
    }                                                                                          \
    TEST_P(SUITE##DependencyUsm, RealDoublePrecision) {                                        \
        CHECK_DOUBLE_ON_DEVICE(GetParam());                                                    \
        test_log::padding = "[          ] ";                                                   \
        EXPECT_TRUEORSKIP(                                                                     \
            dependency_controller.run(::usm_dependency<RealDoublePrecisionUsm>, *GetParam())); \
    }

#define DEFINE_TEST_DEPENDENCY_COMPLEX(SUITE)                                                     \
    TEST_P(SUITE##DependencyUsm, ComplexSinglePrecision) {                                        \
        test_log::padding = "[          ] ";                                                      \
        EXPECT_TRUEORSKIP(                                                                        \
            dependency_controller.run(::usm_dependency<ComplexSinglePrecisionUsm>, *GetParam())); \
    }                                                                                             \
    TEST_P(SUITE##DependencyUsm, ComplexDoublePrecision) {                                        \
        CHECK_DOUBLE_ON_DEVICE(GetParam());                                                       \
        test_log::padding = "[          ] ";                                                      \
        EXPECT_TRUEORSKIP(                                                                        \
            dependency_controller.run(::usm_dependency<ComplexDoublePrecisionUsm>, *GetParam())); \
    }\
