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
*       This example demonstrates use of oneapi::math::lapack::getrf and
*       oneapi::math::lapack::getrs to perform LU factorization and compute
*       the solution on both an Intel CPU device and NVIDIA GPU device.
*
*       This example demonstrates only single precision (float) data type
*       for matrix data
*
*******************************************************************************/

// STL includes
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
// Main example for LU consisting of initialization of
// a general dense A matrix.
// Then the LU factorization
// A = P * L * U
// is performed followed by solving a system of linear
// equations using the computed LU factorization, with
// multiple right-hand sides.
// Finally the results are post processed.
//

void run_getrs_example(const sycl::device& cpu_device, const sycl::device& gpu_device) {
    // Matrix sizes and leading dimensions
    std::int64_t m = 23;
    std::int64_t n = 23;
    std::int64_t nrhs = 23;
    std::int64_t lda = 32;
    std::int64_t ldb = 32;
    std::int64_t A_size = n * lda;
    std::int64_t B_size = nrhs * ldb;
    std::int64_t ipiv_size = n;
    oneapi::math::transpose trans = oneapi::math::transpose::nontrans;

    // Catch asynchronous exceptions for CPU and GPU
    auto cpu_error_handler = [&](sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (oneapi::math::lapack::exception const& e) {
                // Handle LAPACK related exceptions that happened during asynchronous call
                std::cerr
                    << "Caught asynchronous LAPACK exception on CPU device during GETRF or GETRS:"
                    << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
                std::cerr << "\tinfo: " << e.info() << std::endl;
            }
            catch (sycl::exception const& e) {
                // Handle not LAPACK related exceptions that happened during asynchronous call
                std::cerr
                    << "Caught asynchronous SYCL exception on CPU device during GETRF or GETRS:"
                    << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };
    auto gpu_error_handler = [&](sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (oneapi::math::lapack::exception const& e) {
                // Handle LAPACK related exceptions that happened during asynchronous call
                std::cerr
                    << "Caught asynchronous LAPACK exception on GPU device during GETRF or GETRS:"
                    << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
                std::cerr << "\tinfo: " << e.info() << std::endl;
            }
            catch (sycl::exception const& e) {
                // Handle not LAPACK related exceptions that happened during asynchronous call
                std::cerr
                    << "Caught asynchronous SYCL exception on GPU device during GETRF or GETRS:"
                    << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    //
    // Data preparation on host
    //
    std::vector<float> A(A_size);
    std::vector<float> B(B_size);
    std::vector<float> result_cpu(B_size);
    std::vector<float> result_gpu(B_size);
    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);
    std::fill(result_cpu.begin(), result_cpu.end(), 0);
    std::fill(result_gpu.begin(), result_gpu.end(), 0);

    rand_matrix(A, trans, m, n, lda);
    rand_matrix(B, trans, n, nrhs, ldb);

    //
    // Preparation on CPU
    //
    sycl::queue cpu_queue(cpu_device, cpu_error_handler);
    sycl::context cpu_context = cpu_queue.get_context();
    sycl::event cpu_getrf_done;
    sycl::event cpu_getrs_done;

    float* cpu_A = sycl::malloc_device<float>(A_size * sizeof(float), cpu_queue);
    float* cpu_B = sycl::malloc_device<float>(B_size * sizeof(float), cpu_queue);
    std::int64_t* cpu_ipiv =
        sycl::malloc_device<std::int64_t>(ipiv_size * sizeof(std::int64_t), cpu_queue);

    std::int64_t cpu_getrf_scratchpad_size = oneapi::math::lapack::getrf_scratchpad_size<float>(
        oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ cpu_queue }, m, n, lda);
    std::int64_t cpu_getrs_scratchpad_size = oneapi::math::lapack::getrs_scratchpad_size<float>(
        oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ cpu_queue }, trans, n, nrhs,
        lda, ldb);
    float* cpu_getrf_scratchpad = sycl::malloc_device<float>(
        cpu_getrf_scratchpad_size * sizeof(float), cpu_device, cpu_context);
    float* cpu_getrs_scratchpad = sycl::malloc_device<float>(
        cpu_getrs_scratchpad_size * sizeof(float), cpu_device, cpu_context);
    if (!cpu_A || !cpu_B || !cpu_ipiv || !cpu_getrf_scratchpad || !cpu_getrs_scratchpad) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }

    // copy data from host to CPU device
    cpu_queue.memcpy(cpu_A, A.data(), A_size * sizeof(float)).wait();
    cpu_queue.memcpy(cpu_B, B.data(), B_size * sizeof(float)).wait();

    //
    // Preparation on GPU
    //
    sycl::queue gpu_queue(gpu_device, gpu_error_handler);
    sycl::context gpu_context = gpu_queue.get_context();
    sycl::event gpu_getrf_done;
    sycl::event gpu_getrs_done;

    float* gpu_A = sycl::malloc_device<float>(A_size * sizeof(float), gpu_queue);
    float* gpu_B = sycl::malloc_device<float>(B_size * sizeof(float), gpu_queue);
    std::int64_t* gpu_ipiv =
        sycl::malloc_device<std::int64_t>(ipiv_size * sizeof(std::int64_t), gpu_queue);

    std::int64_t gpu_getrf_scratchpad_size = oneapi::math::lapack::getrf_scratchpad_size<float>(
        oneapi::math::backend_selector<oneapi::math::backend::cusolver>{ gpu_queue }, m, n, lda);
    std::int64_t gpu_getrs_scratchpad_size = oneapi::math::lapack::getrs_scratchpad_size<float>(
        oneapi::math::backend_selector<oneapi::math::backend::cusolver>{ gpu_queue }, trans, n,
        nrhs, lda, ldb);
    float* gpu_getrf_scratchpad = sycl::malloc_device<float>(
        gpu_getrf_scratchpad_size * sizeof(float), gpu_device, gpu_context);
    float* gpu_getrs_scratchpad = sycl::malloc_device<float>(
        gpu_getrs_scratchpad_size * sizeof(float), gpu_device, gpu_context);
    if (!gpu_A || !gpu_B || !gpu_ipiv || !gpu_getrf_scratchpad) {
        throw std::runtime_error("Failed to allocate USM memory.");
    }

    // copy data from host to CPU device
    gpu_queue.memcpy(gpu_A, A.data(), A_size * sizeof(float)).wait();
    gpu_queue.memcpy(gpu_B, B.data(), B_size * sizeof(float)).wait();

    //
    // Execute on CPU and GPU devices
    //

    cpu_getrf_done = oneapi::math::lapack::getrf(
        oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ cpu_queue }, m, n, cpu_A,
        lda, cpu_ipiv, cpu_getrf_scratchpad, cpu_getrf_scratchpad_size);
    cpu_getrs_done = oneapi::math::lapack::getrs(
        oneapi::math::backend_selector<oneapi::math::backend::mklcpu>{ cpu_queue }, trans, n, nrhs,
        cpu_A, lda, cpu_ipiv, cpu_B, ldb, cpu_getrs_scratchpad, cpu_getrs_scratchpad_size,
        { cpu_getrf_done });
    gpu_getrf_done = oneapi::math::lapack::getrf(
        oneapi::math::backend_selector<oneapi::math::backend::cusolver>{ gpu_queue }, m, n, gpu_A,
        lda, gpu_ipiv, gpu_getrf_scratchpad, gpu_getrf_scratchpad_size);
    gpu_getrs_done = oneapi::math::lapack::getrs(
        oneapi::math::backend_selector<oneapi::math::backend::cusolver>{ gpu_queue }, trans, n,
        nrhs, gpu_A, lda, gpu_ipiv, gpu_B, ldb, gpu_getrs_scratchpad, gpu_getrs_scratchpad_size,
        { gpu_getrf_done });

    // Wait until calculations are done
    cpu_queue.wait_and_throw();
    gpu_queue.wait_and_throw();

    //
    // Post Processing
    //
    // copy data from CPU device back to host
    cpu_queue.memcpy(result_cpu.data(), cpu_B, B_size * sizeof(float)).wait_and_throw();

    // copy data from GPU device back to host
    gpu_queue.memcpy(result_gpu.data(), gpu_B, B_size * sizeof(float)).wait_and_throw();

    // Print results
    std::cout << "\n\t\tGETRF and GETRS parameters:" << std::endl;
    std::cout << "\t\t\ttrans = "
              << (trans == oneapi::math::transpose::nontrans
                      ? "nontrans"
                      : (trans == oneapi::math::transpose::trans ? "trans" : "conjtrans"))
              << std::endl;
    std::cout << "\t\t\tm = " << m << ", n = " << n << ", nrhs = " << nrhs << std::endl;
    std::cout << "\t\t\tlda = " << lda << ", ldb = " << ldb << std::endl;

    std::cout << "\n\t\tOutputting 2x2 block of A and X matrices:" << std::endl;
    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(A.data(), lda, "A");

    // output the top 2x2 block of X matrix from CPU
    print_2x2_matrix_values(result_cpu.data(), ldb, "(CPU) X");

    // output the top 2x2 block of X matrix from GPU
    print_2x2_matrix_values(result_gpu.data(), ldb, "(GPU) X");

    sycl::free(gpu_getrs_scratchpad, gpu_queue);
    sycl::free(gpu_getrf_scratchpad, gpu_queue);
    sycl::free(gpu_ipiv, gpu_queue);
    sycl::free(gpu_B, gpu_queue);
    sycl::free(gpu_A, gpu_queue);

    sycl::free(cpu_getrs_scratchpad, cpu_queue);
    sycl::free(cpu_getrf_scratchpad, cpu_queue);
    sycl::free(cpu_ipiv, cpu_queue);
    sycl::free(cpu_B, cpu_queue);
    sycl::free(cpu_A, cpu_queue);
}

//
// Description of example setup, apis used and supported floating point type precisions
//

void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << "# LU Factorization and Solve Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Computes LU Factorization A = P * L * U" << std::endl;
    std::cout << "# and uses it to solve for X in a system of linear equations:" << std::endl;
    std::cout << "#   AX = B" << std::endl;
    std::cout << "# where A is a general dense matrix and B is a matrix whose columns" << std::endl;
    std::cout << "# are the right-hand sides for the systems of equations." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   getrf and getrs" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Running on both Intel CPU and NVIDIA GPU devices" << std::endl;
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
            std::cerr << "FAILED: NVIDIA GPU device not found." << std::endl;
            return 1;
        }
        std::cout << "Running LAPACK GETRS USM example" << std::endl;
        std::cout << "Running with single precision real data type on:" << std::endl;
        std::cout << "\tCPU device :" << cpu_dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "\tGPU device :" << gpu_dev.get_info<sycl::info::device::name>() << std::endl;

        run_getrs_example(cpu_dev, gpu_dev);
        std::cout << "LAPACK GETRS USM example ran OK on MKLCPU and CUSOLVER" << std::endl;
    }
    catch (oneapi::math::lapack::exception const& e) {
        // Handle LAPACK related exceptions that happened during synchronous call
        std::cerr << "Caught synchronous LAPACK exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tinfo: " << e.info() << std::endl;
        return 1;
    }
    catch (sycl::exception const& e) {
        // Handle not LAPACK related exceptions that happened during synchronous call
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
