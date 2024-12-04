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

// stl includes
#include <iostream>
#include <cstdint>

// oneMath/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math.hpp"

void run_example(const sycl::device& dev) {
    constexpr std::size_t N = 16;

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    std::cout << "DFT example run_time dispatch" << std::endl;

    sycl::queue sycl_queue(dev, exception_handler);
    auto x_usm = sycl::malloc_shared<float>(N * 2, sycl_queue);

    // 1. create descriptors
    oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                  oneapi::math::dft::domain::REAL>
        desc(static_cast<std::int64_t>(N));

    // 2. variadic set_value
    desc.set_value(oneapi::math::dft::config_param::NUMBER_OF_TRANSFORMS,
                   static_cast<std::int64_t>(1));
    desc.set_value(oneapi::math::dft::config_param::PLACEMENT,
                   oneapi::math::dft::config_value::INPLACE);

    // 3. commit_descriptor (runtime dispatch)
    desc.commit(sycl_queue);

    // 4. compute_forward / compute_backward (runtime dispatch)
    auto compute_event = oneapi::math::dft::compute_forward(desc, x_usm);

    // Do something with transformed data.
    compute_event.wait();

    // 5. Free USM allocation.
    sycl::free(x_usm, sycl_queue);
}

//
// Description of example setup, APIs used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "########################################################################\n"
                 "# DFT complex in-place forward transform with USM API example:\n"
                 "#\n"
                 "# Using APIs:\n"
                 "#   USM forward complex in-place\n"
                 "#   Run-time dispatch\n"
                 "#\n"
                 "# Using single precision (float) data type\n"
                 "#\n"
                 "# Device will be selected during runtime.\n"
                 "# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify\n"
                 "# available devices\n"
                 "#\n"
                 "########################################################################\n"
              << std::endl;
}

//
// Main entry point for example.
//

int main(int /*argc*/, char** /*argv*/) {
    print_example_banner();

    try {
        sycl::device my_dev((sycl::default_selector_v));

        if (my_dev.is_gpu()) {
            std::cout << "Running DFT complex forward example on GPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        else {
            std::cout << "Running DFT complex forward example on CPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        std::cout << "Running with single precision real data type:" << std::endl;

        run_example(my_dev);
        std::cout << "DFT example ran OK" << std::endl;
    }
    catch (oneapi::math::unimplemented const& e) {
        std::cerr << "Unsupported Configuration:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 0;
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        std::cerr << "Caught std::exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }
    return 0;
}
