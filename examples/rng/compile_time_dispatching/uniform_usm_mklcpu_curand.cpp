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
*       random numbers on a INTEL CPU SYCL device and an NVIDIA GPU SYCL device
*       with Unified Shared Memory(USM) API.
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
void run_uniform_example(const sycl::device& cpu_dev, const sycl::device& gpu_dev) {
    //
    // Initialization
    //
    // example parameters defines
    constexpr std::uint64_t seed = 777;
    constexpr std::size_t n = 1000;
    constexpr std::size_t n_print = 10;
    constexpr std::size_t alignment = 64;

    // Catch asynchronous exceptions for CPU and GPU
    auto cpu_exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception on CPU device during generation:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };
    auto gpu_exception_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                std::cerr << "Caught asynchronous SYCL exception on GPU device during generation:"
                          << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    // set scalar Type values
    float a(0.0);
    float b(10.0);

    // preparation on CPU device and GPU device
    sycl::queue cpu_queue(cpu_dev, cpu_exception_handler);
    sycl::queue gpu_queue(gpu_dev, gpu_exception_handler);
    oneapi::math::rng::default_engine cpu_engine(
        oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ cpu_queue }, seed);
    oneapi::math::rng::default_engine gpu_engine(
        oneapi::math::backend_selector<oneapi::math::backend::curand>{ gpu_queue }, seed);

    oneapi::math::rng::uniform<float> distribution(a, b);

    //
    // Data preparation on host: prepare array for random numbers
    //
    std::vector<float> r_cpu(n);
    std::vector<float> r_gpu(n);
    std::fill(r_cpu.begin(), r_cpu.end(), 0);
    std::fill(r_gpu.begin(), r_gpu.end(), 0);

    //
    // Data preparation on CPU device and GPU device
    //
    float* dev_cpu = sycl::malloc_device<float>(n * sizeof(float), cpu_queue);
    float* dev_gpu = sycl::malloc_device<float>(n * sizeof(float), gpu_queue);
    if (!dev_cpu || !dev_gpu) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }

    //
    // Perform generation on CPU device and GPU device
    //
    sycl::event event_out_cpu;
    sycl::event event_out_gpu;
    event_out_cpu = oneapi::math::rng::generate(distribution, cpu_engine, n, dev_cpu);
    event_out_gpu = oneapi::math::rng::generate(distribution, gpu_engine, n, dev_gpu);
    event_out_cpu.wait_and_throw();
    event_out_gpu.wait_and_throw();

    //
    // Post Processing
    //

    // copy data from CPU device and GPU device back to host
    cpu_queue.memcpy(r_cpu.data(), dev_cpu, n * sizeof(float)).wait_and_throw();
    gpu_queue.memcpy(r_gpu.data(), dev_gpu, n * sizeof(float)).wait_and_throw();

    std::cout << "\t\tgeneration parameters:" << std::endl;
    std::cout << "\t\t\tseed = " << seed << ", a = " << a << ", b = " << b << std::endl;

    std::cout << "\t\tOutput of generator on CPU device:" << std::endl;
    std::cout << "\t\t\tfirst " << n_print << " numbers of " << n << ": " << std::endl;
    for (int i = 0; i < n_print; i++) {
        std::cout << r_cpu.at(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "\t\tOutput of generator on GPU device:" << std::endl;
    std::cout << "\t\t\tfirst " << n_print << " numbers of " << n << ": " << std::endl;
    for (int i = 0; i < n_print; i++) {
        std::cout << r_gpu.at(i) << " ";
    }
    std::cout << std::endl;

    sycl::free(dev_gpu, gpu_queue);
    sycl::free(dev_cpu, cpu_queue);
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
    std::cout << "# Running on both Intel CPU and Nvidia GPU devices" << std::endl;
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
        sycl::device cpu_dev((sycl::cpu_selector()));
        sycl::device gpu_dev((sycl::gpu_selector()));

        unsigned int vendor_id = gpu_dev.get_info<sycl::info::device::vendor_id>();
        if (vendor_id != NVIDIA_ID) {
            std::cerr << "FAILED: NVIDIA GPU device not found" << std::endl;
            return 1;
        }
        std::cout << "Running RNG uniform usm example" << std::endl;
        std::cout << "Running with single precision real data type:" << std::endl;
        std::cout << "\tCPU device: " << cpu_dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "\tGPU device: " << gpu_dev.get_info<sycl::info::device::name>() << std::endl;

        run_uniform_example(cpu_dev, gpu_dev);
        std::cout
            << "Random number generator example with uniform distribution ran OK on MKLCPU and CURAND"
            << std::endl;
    }
    catch (sycl::exception const& e) {
        std::cerr << "Caught synchronous SYCL exception during generation:" << std::endl;
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
