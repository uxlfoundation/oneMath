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

/*
*
*  Content:
*       This example demonstrates use of the DPCPP API
*       oneapi::math::dft to perform a complex in-place forward transform
*       using unified shared memory on a SYCL device (CPU, GPU) with the
*       SYCL-only "portFFT" backend and compile-time dispatching.
*
*       Note that the portFFT backend only supports the COMPLEX domain.
*
*       This example demonstrates only single precision (float) data type.
*
*
*******************************************************************************/

// STL includes
#include <iostream>
#include <complex>

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
                std::cerr << "Caught asynchronous SYCL exception during execution:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    sycl::queue sycl_queue(dev, exception_handler);
    auto in_out = sycl::malloc_shared<std::complex<float>>(N, sycl_queue);

    // Initialize input data
    for (std::size_t i = 0; i < N; ++i) {
        in_out[i] = { static_cast<float>(i), static_cast<float>(-i) };
    }

    // 1. create descriptor
    oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                  oneapi::math::dft::domain::COMPLEX>
        desc(static_cast<std::int64_t>(N));

    // 2. variadic set_value
    desc.set_value(oneapi::math::dft::config_param::PLACEMENT,
                   oneapi::math::dft::config_value::INPLACE);
    desc.set_value(oneapi::math::dft::config_param::NUMBER_OF_TRANSFORMS,
                   static_cast<std::int64_t>(1));

    // 3. commit_descriptor (compile-time portFFT)
    desc.commit(oneapi::math::backend_selector<oneapi::math::backend::portfft>{ sycl_queue });

    // 4. compute_forward (portFFT)
    oneapi::math::dft::compute_forward<decltype(desc), std::complex<float>>(desc, in_out);

    sycl_queue.wait_and_throw();

    sycl::free(in_out, sycl_queue);
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "\n"
                 "########################################################################\n"
                 "# Complex in-place forward transform for USM API's example:\n"
                 "#\n"
                 "# Using APIs:\n"
                 "#   Compile-time dispatch API\n"
                 "#   USM forward complex in-place\n"
                 "#\n"
                 "# Using single precision (float) data type\n"
                 "#\n"
                 "# Running on a SYCL device with the portFFT backend.\n"
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
        sycl::device dev((sycl::default_selector_v));

        if (dev.is_gpu()) {
            std::cout << "Running DFT complex forward example on GPU device" << std::endl;
        }
        else {
            std::cout << "Running DFT complex forward example on CPU device" << std::endl;
        }
        std::cout << "Device name is: " << dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Using compile-time dispatch API with portFFT." << std::endl;
        std::cout << "Running with single precision real data type:" << std::endl;

        run_example(dev);
        std::cout << "DFT Complex USM example ran OK on portFFT" << std::endl;
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
