/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <gtest/gtest.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <string>
#include "test_helper.hpp"
#include "oneapi/math/detail/config.hpp"
#include "oneapi/math.hpp"

#define MAX_STR 128

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestCase;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

std::vector<sycl::device*> devices;

std::string gtestInFile;

namespace {
// Provides alternative output mode which produces minimal amount of
// information about tests.
class TersePrinter : public EmptyTestEventListener {
private:
    // Called before any test activity starts.
    void OnTestProgramStart(const UnitTest& /* unit_test */) override {}

    // Called after all test activities have ended.
    void OnTestProgramEnd(const UnitTest& unit_test) override {
        fprintf(stdout, "TEST %s\n", unit_test.Passed() ? "PASSED" : "FAILED");
        fflush(stdout);
    }

    // Called before a test starts.
    void OnTestStart(const TestInfo& test_info) override {
        fprintf(stdout, "*** Test %s.%s starting.\n", test_info.test_case_name(), test_info.name());
        fflush(stdout);
    }

    // Called after a failed assertion or a SUCCEED() invocation.
    void OnTestPartResult(const TestPartResult& test_part_result) override {
        const char* file_name = test_part_result.file_name();
        fprintf(stdout, "%s in %s:%d\n%s\n", test_part_result.failed() ? "*** Failure" : "Success",
                file_name ? file_name : "unknown file", test_part_result.line_number(),
                test_part_result.summary());
        fflush(stdout);
    }

    // Called after a test ends.
    void OnTestEnd(const TestInfo& test_info) override {
        fprintf(stdout, "*** Test %s.%s ending.\n", test_info.test_case_name(), test_info.name());
        fflush(stdout);
    }
}; // class TersePrinter

} // anonymous namespace

void print_error_code(sycl::exception const& e) {
#ifdef __HIPSYCL__
    std::cout << "Backend status: " << e.code() << std::endl;
#else
    std::cout << "OpenCL status: " << e.code() << std::endl;
#endif
}

int main(int argc, char** argv) {
    std::set<std::string> unique_devices;
    std::vector<sycl::device> local_devices;

    auto platforms = sycl::platform::get_platforms();
    for (auto plat : platforms) {
#ifdef __HIPSYCL__
        if (!plat.is_host()) {
#endif
            auto plat_devs = plat.get_devices();
            for (auto dev : plat_devs) {
                try {
                    unsigned int vendor_id =
                        static_cast<unsigned int>(dev.get_info<sycl::info::device::vendor_id>());
                    /* Do not test for OpenCL backend on Intel GPU */
                    if (dev.is_gpu() &&
                        plat.get_info<sycl::info::platform::name>().find("OpenCL") !=
                            std::string::npos &&
                        vendor_id == INTEL_ID)
                        continue;
                    if (unique_devices.find(dev.get_info<sycl::info::device::name>()) ==
                        unique_devices.end()) {
                        unique_devices.insert(dev.get_info<sycl::info::device::name>());
#if !defined(ONEMATH_ENABLE_MKLCPU_BACKEND) &&             \
    !defined(ONEMATH_ENABLE_PORTBLAS_BACKEND_INTEL_CPU) && \
    !defined(ONEMATH_ENABLE_PORTFFT_BACKEND) && !defined(ONEMATH_ENABLE_NETLIB_BACKEND)
                        if (dev.is_cpu())
                            continue;
#endif
#if !defined(ONEMATH_ENABLE_MKLGPU_BACKEND) &&             \
    !defined(ONEMATH_ENABLE_PORTBLAS_BACKEND_INTEL_GPU) && \
    !defined(ONEMATH_ENABLE_PORTFFT_BACKEND)
                        if (dev.is_gpu() && vendor_id == INTEL_ID)
                            continue;
#endif
#if !defined(ONEMATH_ENABLE_CUBLAS_BACKEND) && !defined(ONEMATH_ENABLE_CURAND_BACKEND) && \
    !defined(ONEMATH_ENABLE_CUSOLVER_BACKEND) &&                                          \
    !defined(ONEMATH_ENABLE_PORTBLAS_BACKEND_NVIDIA_GPU) &&                               \
    !defined(ONEMATH_ENABLE_CUFFT_BACKEND) && !defined(ONEMATH_ENABLE_PORTFFT_BACKEND) && \
    !defined(ONEMATH_ENABLE_CUSPARSE_BACKEND)
                        if (dev.is_gpu() && vendor_id == NVIDIA_ID)
                            continue;
#endif
#if !defined(ONEMATH_ENABLE_ROCBLAS_BACKEND) && !defined(ONEMATH_ENABLE_ROCRAND_BACKEND) && \
    !defined(ONEMATH_ENABLE_ROCSOLVER_BACKEND) &&                                           \
    !defined(ONEMATH_ENABLE_PORTBLAS_BACKEND_AMD_GPU) &&                                    \
    !defined(ONEMATH_ENABLE_ROCFFT_BACKEND) && !defined(ONEMATH_ENABLE_PORTFFT_BACKEND)
                        if (dev.is_gpu() && vendor_id == AMD_ID)
                            continue;
#endif
                        // clang-format off
#ifdef __HIPSYCL__
                        if (dev.is_accelerator())
#else
                        if (!dev.is_accelerator())
// clang-format on
#endif
                            local_devices.push_back(dev);
                    }
                }
                catch (std::exception const& e) {
                    std::cout << "Exception while accessing device: " << e.what() << "\n";
                }
            }
#ifdef __HIPSYCL__
        }
#endif
    }

#define GET_NAME(d) (d).template get_info<sycl::info::device::name>()
    for (auto& local_dev : local_devices) {
        // Test only unique devices
        if (std::find_if(devices.begin(), devices.end(), [&](sycl::device* dev) {
                return GET_NAME(*dev) == GET_NAME(local_dev);
            }) == devices.end())
            devices.push_back(&local_dev);
    }

    // start Google Test pickup and output
    testing::InitGoogleTest(&argc, argv);

    bool terse_output = false;
    if (argc > 1 && strcmp(argv[1], "--terse_output") == 0)
        terse_output = true;
    else
        printf("%s\n",
               "Run this program with --terse_output to change the way it prints its output.");

    for (int i = 0; i < argc; i++) {
        if (strncmp(argv[i], "--input_file=", 13) == 0) {
            std::string tmp(argv[i]);
            gtestInFile = tmp.substr(13);
            break;
        }
    }

    UnitTest& unit_test = *UnitTest::GetInstance();

    // If we are given the --terse_output command line flag, suppresses the
    // standard output and attaches own result printer.
    if (terse_output) {
        TestEventListeners& listeners = unit_test.listeners();

        // Removes the default console output listener from the list so it will
        // not receive events from Google Test and won't print any output. Since
        // this operation transfers ownership of the listener to the caller we
        // have to delete it as well.
        delete listeners.Release(listeners.default_result_printer());

        // Adds the custom output listener to the list. It will now receive
        // events from Google Test and print the alternative output. We don't
        // have to worry about deleting it since Google Test assumes ownership
        // over it after adding it to the list.
        listeners.Append(new TersePrinter);
    }
    int ret_val = RUN_ALL_TESTS();

    // This is an example of using the UnitTest reflection API to inspect test
    // results. Here we discount failures from the tests we expected to fail.
    int unexpectedly_failed_tests = 0;
    for (int i = 0; i < unit_test.total_test_case_count(); ++i) {
        const TestCase& test_case = *unit_test.GetTestCase(i);
        for (int j = 0; j < test_case.total_test_count(); ++j) {
            const TestInfo& test_info = *test_case.GetTestInfo(j);
            // Counts failed tests that were not meant to fail (those without
            // 'Fails' in the name).
            if (test_info.result()->Failed() && strcmp(test_info.name(), "Fails") != 0) {
                unexpectedly_failed_tests++;
            }
        }
    }

    // Test that were meant to fail should not affect the test program outcome.
    if (unexpectedly_failed_tests == 0)
        ret_val = 0;

    return ret_val;
}
