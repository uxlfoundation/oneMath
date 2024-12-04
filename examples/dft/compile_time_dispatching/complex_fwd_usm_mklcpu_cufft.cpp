/*******************************************************************************
* Copyright 2024 Intel Corporation
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

// STL includes
#include <iostream>

// oneMath/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/math.hpp"
#include <complex>

void run_example(const sycl::device& cpu_device, const sycl::device& gpu_device) {
    constexpr std::size_t N = 10;

    // Catch asynchronous exceptions for cpu
    auto cpu_error_handler = [&](sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                // Handle not dft related exceptions that happened during asynchronous call
                std::cerr << "Caught asynchronous SYCL exception on CPU device during execution:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };
    // Catch asynchronous exceptions for gpu
    auto gpu_error_handler = [&](sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                // Handle not dft related exceptions that happened during asynchronous call
                std::cerr << "Caught asynchronous SYCL exception on GPU device during execution:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    // Preparation CPU device and GPU device
    sycl::queue cpu_queue(cpu_device, cpu_error_handler);
    sycl::queue gpu_queue(gpu_device, gpu_error_handler);

    // allocate on CPU device and GPU device
    auto cpu_input_data = sycl::malloc_shared<std::complex<float>>(N, cpu_queue);
    auto cpu_output_data = sycl::malloc_shared<std::complex<float>>(N, cpu_queue);

    auto gpu_input_data = sycl::malloc_shared<std::complex<float>>(N, gpu_queue);
    auto gpu_output_data = sycl::malloc_shared<std::complex<float>>(N, gpu_queue);

    // Initialize input data
    for (std::size_t i = 0; i < N; ++i) {
        cpu_input_data[i] = { static_cast<float>(i), static_cast<float>(-i) };
        gpu_input_data[i] = { static_cast<float>(i), static_cast<float>(-i) };
    }

    // enabling
    // 1. create descriptors
    oneapi::math::dft::descriptor<oneapi::math::dft::precision::SINGLE,
                                  oneapi::math::dft::domain::COMPLEX>
        desc(static_cast<std::int64_t>(N));

    // 2. variadic set_value
    desc.set_value(oneapi::math::dft::config_param::PLACEMENT,
                   oneapi::math::dft::config_value::NOT_INPLACE);
    desc.set_value(oneapi::math::dft::config_param::NUMBER_OF_TRANSFORMS,
                   static_cast<std::int64_t>(1));

    // 3a. commit_descriptor (compile_time MKLCPU)
    desc.commit(oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ cpu_queue });

    // 4a. compute_forward / compute_backward (MKLCPU)
    oneapi::math::dft::compute_forward<decltype(desc), std::complex<float>, std::complex<float>>(
        desc, cpu_input_data, cpu_output_data);

    // 3b. commit_descriptor (compile_time cuFFT)
    desc.commit(oneapi::math::backend_selector<oneapi::math::backend::cufft>{ gpu_queue });

    // 4b. compute_forward / compute_backward (cuFFT)
    oneapi::math::dft::compute_forward<decltype(desc), std::complex<float>, std::complex<float>>(
        desc, gpu_input_data, gpu_output_data);

    cpu_queue.wait_and_throw();
    gpu_queue.wait_and_throw();

    sycl::free(cpu_input_data, cpu_queue);
    sycl::free(gpu_input_data, gpu_queue);
    sycl::free(cpu_output_data, cpu_queue);
    sycl::free(gpu_output_data, gpu_queue);
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "\n"
                 "########################################################################\n"
                 "# Complex out-of-place forward transform for USM API's example:\n"
                 "#\n"
                 "# Using APIs:\n"
                 "#   Compile-time dispatch API\n"
                 "#   USM forward complex out-of-place\n"
                 "#\n"
                 "# Using single precision (float) data type\n"
                 "#\n"
                 "# Running on both Intel CPU and NVIDIA GPU devices.\n"
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
        sycl::device cpu_device((sycl::cpu_selector_v));
        sycl::device gpu_device((sycl::gpu_selector_v));

        unsigned int vendor_id = gpu_device.get_info<sycl::info::device::vendor_id>();
        if (vendor_id != NVIDIA_ID) {
            std::cerr << "FAILED: NVIDIA GPU device not found" << std::endl;
            return 1;
        }

        std::cout << "Running DFT Complex forward out-of-place usm example" << std::endl;
        std::cout << "Using compile-time dispatch API with MKLCPU and cuFFT." << std::endl;
        std::cout << "Running with single precision real data type on:" << std::endl;
        std::cout << "\tCPU device: " << cpu_device.get_info<sycl::info::device::name>()
                  << std::endl;
        std::cout << "\tGPU device :" << gpu_device.get_info<sycl::info::device::name>()
                  << std::endl;

        run_example(cpu_device, gpu_device);
        std::cout << "DFT Complex USM example ran OK on MKLCPU and CUFFT" << std::endl;
    }
    catch (sycl::exception const& e) {
        // Handle not dft related exceptions that happened during synchronous call
        std::cerr << "Caught synchronous SYCL exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        // Handle not SYCL related exceptions that happened during synchronous call
        std::cerr << "Caught synchronous std::exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }

    return 0;
}
