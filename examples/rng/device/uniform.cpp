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
*       This example demonstrates usage of oneapi::mkl::rng::device::mcg59
*       random number generator to produce random
*       numbers using unifrom distribution on a SYCL device (CPU, GPU).
*
*******************************************************************************/

// stl includes
#include <iostream>
#include <vector>

// oneMKL/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/rng/device.hpp"

#include "rng_example_helper.hpp"

bool isDoubleSupported(sycl::device my_dev) {
    return my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0;
}

// example parameters
constexpr std::uint64_t seed = 777;
constexpr std::size_t n = 1024;
constexpr int n_print = 10;

//
// example show usage of rng device functionality, which can be called from both
// host and device sides with scalar and vector generation
//
template <typename Type, std::int32_t VecSize>
int run_example(sycl::queue& queue) {
    if (VecSize == 1) {
        std::cout << "\tRunning scalar example" << std::endl;
    }
    else {
        std::cout << "\tRunning vector example with " << VecSize << " vector size" << std::endl;
    }
    // prepare array for random numbers
    std::vector<Type> r_dev(n);

    // submit a kernel to generate on device
    {
        sycl::buffer<Type> r_buf(r_dev.data(), r_dev.size());

        try {
            queue.submit([&](sycl::handler& cgh) {
                sycl::accessor r_acc(r_buf, cgh, sycl::write_only);
                cgh.parallel_for(sycl::range<1>(n / VecSize), [=](sycl::item<1> item) {
                    size_t item_id = item.get_id(0);
                    oneapi::mkl::rng::device::mcg59<VecSize> engine(seed, item_id * VecSize);
                    oneapi::mkl::rng::device::uniform<Type> distr;

                    auto res = oneapi::mkl::rng::device::generate(distr, engine);
                    if constexpr (VecSize == 1) {
                        r_acc[item_id] = res;
                    }
                    else {
                        res.store(item_id, get_multi_ptr(r_acc));
                    }
                });
            });
            queue.wait_and_throw();
        }
        catch (sycl::exception const& e) {
            std::cout << "\t\tSYCL exception\n" << e.what() << std::endl;
            return 1;
        }

        std::cout << "\t\tOutput of generator:" << std::endl;

        auto r_acc = sycl::host_accessor(r_buf, sycl::read_only);
        std::cout << "first " << n_print << " numbers of " << n << ": " << std::endl;
        for (int i = 0; i < n_print; i++) {
            std::cout << r_acc[i] << " ";
        }
        std::cout << std::endl;
    } // buffer life-time ends

    // compare results with host-side generation
    oneapi::mkl::rng::device::mcg59<1> engine(seed);
    oneapi::mkl::rng::device::uniform<Type> distr;

    int err = 0;
    Type res_host;
    for (std::size_t i = 0; i < n; i++) {
        res_host = oneapi::mkl::rng::device::generate(distr, engine);
        if (res_host != r_dev[i]) {
            std::cout << "error in " << i << " element " << res_host << " " << r_dev[i]
                      << std::endl;
            err++;
        }
    }
    return err;
}

//
// description of example setup, APIs used
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << "# Generate uniformly distributed random numbers example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using APIs:" << std::endl;
    std::cout << "# mcg59 uniform" << std::endl;
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
            std::cout << "Running RNG uniform usm example on GPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }
        else {
            std::cout << "Running RNG uniform usm example on CPU device" << std::endl;
            std::cout << "Device name is: " << my_dev.get_info<sycl::info::device::name>()
                      << std::endl;
        }

        sycl::queue queue(my_dev, exception_handler);

        std::cout << "\n\tRunning with single precision real data type:" << std::endl;
        if (run_example<float, 1>(queue) || run_example<float, 4>(queue)) {
            std::cout << "FAILED" << std::endl;
            return 1;
        }
        if (isDoubleSupported(my_dev)) {
            std::cout << "\n\tRunning with double precision real data type:" << std::endl;
            if (run_example<double, 1>(queue) || run_example<double, 4>(queue)) {
                std::cout << "FAILED" << std::endl;
                return 1;
            }
        }
        else {
            std::cout << "Double precision is not supported for this device" << std::endl;
        }
        std::cout << "\n\tRunning with integer data type:" << std::endl;
        if (run_example<std::int32_t, 1>(queue) || run_example<std::int32_t, 4>(queue)) {
            std::cout << "FAILED" << std::endl;
            return 1;
        }
        std::cout << "\n\tRunning with unsigned integer data type:" << std::endl;
        if (run_example<std::uint32_t, 1>(queue) || run_example<std::uint32_t, 4>(queue)) {
            std::cout << "FAILED" << std::endl;
            return 1;
        }

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
