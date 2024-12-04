/*******************************************************************************
* Copyright 2022 Intel Corporation
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

/*
*
*  Content:
*       This example demonstrates use of DPC++ API oneapi::math::rng::uniform distribution
*       with oneapi::math::rng::philox4x32x10 random number generator to produce
*       random numbers on a SYCL device (HOST, CPU, GPU) that is selected
*       during runtime with Unified Shared Memory(USM) API.
*
*       This example demonstrates only single precision (float) data type
*       for random numbers
*
*******************************************************************************/

// stl includes
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// oneMath/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/math.hpp"

// local includes
#include "example_helper.hpp"

//
// Main example for Uniform random number generation consisting of
// initialization of random number engine philox4x32x10 object, distribution
// object. Then random number generation performed and
// the output is post-processed and validated.
//
void run_uniform_example(const sycl::device& dev) {
    //
    // Initialization
    //
    // example parameters defines
    constexpr std::uint64_t seed = 777;
    constexpr std::size_t n = 1000;
    constexpr std::size_t n_print = 10;
    constexpr std::size_t alignment = 64;

    // Catch asynchronous exceptions
    auto exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception during generation:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    sycl::queue queue(dev, exception_handler);

    // set scalar Type values
    float a(0.0);
    float b(10.0);

    oneapi::math::rng::default_engine engine(queue, seed);
    oneapi::math::rng::uniform<float> distribution(a, b);

    //
    // Data preparation on host: prepare array for random numbers
    //
    std::vector<float> r(n);

    // Data preparation on selected device
    float* dev_r = sycl::malloc_device<float>(n * sizeof(float), queue);
    if (!dev_r) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }

    //
    // Perform generation on device
    //
    sycl::event event_out;
    event_out = oneapi::math::rng::generate(distribution, engine, n, dev_r);
    event_out.wait_and_throw();

    //
    // Post Processing
    //

    // copy data from device back to host
    queue.memcpy(r.data(), dev_r, n * sizeof(float)).wait_and_throw();

    std::cout << "\t\tgeneration parameters:" << std::endl;
    std::cout << "\t\t\tseed = " << seed << ", a = " << a << ", b = " << b << std::endl;

    std::cout << "\t\tOutput of generator:" << std::endl;
    std::cout << "\t\t\tfirst " << n_print << " numbers of " << n << ": " << std::endl;
    for (int i = 0; i < n_print; i++) {
        std::cout << r.at(i) << " ";
    }
    std::cout << std::endl;

    sycl::free(dev_r, queue);
}

//
// Description of example setup, APIs used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout
        << "# Generate uniformly distributed random numbers with philox4x32x10\n# generator example: "
        << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using APIs:" << std::endl;
    std::cout << "#   default_engine uniform" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Device will be selected during runtime." << std::endl;
    std::cout << "# The environment variable ONEAPI_DEVICE_SELECTOR can be used to specify"
              << std::endl;
    std::cout << "# available devices" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example.
//

int main(int argc, char** argv) {
    print_example_banner();

    try {
        sycl::device my_dev = sycl::device();

        if (my_dev.is_gpu()) {
            std::cout << "Running RNG uniform usm example on GPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        else {
            std::cout << "Running RNG uniform usm example on CPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        std::cout << "Running with single precision real data type:" << std::endl;

        run_uniform_example(my_dev);
        std::cout << "Random number generator with uniform distribution ran OK" << std::endl;
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        std::cerr << "Caught std::exception during generation:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }
    return 0;
}
