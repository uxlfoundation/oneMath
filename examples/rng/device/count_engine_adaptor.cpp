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

/*
*
*  Content:
*       This example demonstrates usage of oneapi::math::rng::device::count_engine_adaptor
*       to produce random numbers using beta distribution on a SYCL device (CPU, GPU).
*
*******************************************************************************/

// stl includes
#include <iostream>
#include <vector>

// oneMath/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/math/rng/device.hpp"

#include "rng_example_helper.hpp"

bool isDoubleSupported(sycl::device my_dev) {
    return my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0;
}

// example parameters
constexpr std::uint64_t seed = 777;
constexpr std::size_t n_per_item = 20;
constexpr std::size_t n = 1024 * n_per_item;
constexpr int n_print = 10;

namespace rng_device = oneapi::math::rng::device;

template <typename Type>
int run_example(sycl::queue& queue) {
    // prepare array for random numbers
    using allocator_t = sycl::usm_allocator<Type, sycl::usm::alloc::shared>;
    allocator_t allocator(queue);

    std::vector<Type, allocator_t> average_vec(n, allocator);
    std::vector<Type, allocator_t> r_count_vec(n / n_per_item, allocator);
    Type* average = average_vec.data();
    Type* r_count = r_count_vec.data();

    // submit a kernel to generate on device
    try {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::item<1> item) {
            size_t item_id = item.get_id(0);
            rng_device::count_engine_adaptor<rng_device::mcg59<1>> adaptor
                (seed, item_id * n_per_item);
            rng_device::gamma<Type> distr(2.0f, 0.1f, 0.9f);

            Type res(0);
            for(std::size_t i = 0; i < n_per_item; i++) {
                res += rng_device::generate(distr, adaptor);
            }
            average[item_id] = res / n_per_item;
            r_count[item_id] = adaptor.get_count();
        }).wait_and_throw();
    }
    catch (sycl::exception const& e) {
        std::cout << "\t\tSYCL exception\n" << e.what() << std::endl;
        return 1;
    }

    std::cout << "\t\tOutput of generator:" << std::endl;

    std::cout << "first " << n_print << " numbers of " << n << ": " << std::endl;
    for (int i = 0; i < n_print; i++) {
        std::cout << average[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "first " << n_print << " engine calls of " << n << ": " << std::endl;
    for (int i = 0; i < n_print; i++) {
        std::cout << r_count[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

//
// description of example setup, APIs used
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << "# Example to use count_engine_adaptor class: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using APIs:" << std::endl;
    std::cout << "# mcg59 beta" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << std::endl;
}

int main() {
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

    print_example_banner();

    try {
        sycl::device my_dev = sycl::device();

        if (my_dev.is_gpu()) {
            std::cout << "Running RNG count_engine_adaptor example on GPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        else {
            std::cout << "Running RNG count_engine_adaptor example on CPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }

        sycl::queue queue(my_dev, exception_handler);

        std::cout << "\n\tRunning with single precision real data type:" << std::endl;
        if (run_example<float>(queue)) {
            std::cout << "FAILED" << std::endl;
            return 1;
        }

        std::cout << "Random number generator's adaptor ran OK" << std::endl;
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
